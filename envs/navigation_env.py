"""
envs/navigation_env.py
MJX 导航环境（纯函数式，支持 jit/vmap）

迁移自: training/scripts/env.py (NavigationEnv)

设计原则：
  - 完全函数式：所有状态通过 EnvState pytree 传递，无可变属性
  - jax.vmap 兼容：单环境 reset/step 函数可直接 vmap 并行化
  - jax.jit 兼容：所有控制流使用 jax.lax.cond/select，无 Python if

环境概述：
  无人机从随机起点飞向随机目标点（±24m 范围），观测量在目标坐标系中表达，
  PPO 输出 3D 速度指令 → Lee 控制器 → 4 电机推力 → MJX 物理仿真。

观测空间 (8D):
  [rpos_clipped_g(3), distance_2d(1), distance_z(1), vel_g(3)]

动作空间 (3D):
  目标坐标系中的速度指令 ∈ [-action_limit, +action_limit] m/s
"""

import math
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
from typing import NamedTuple, Tuple

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.math_utils import (
    vec_to_new_frame,
    vec_to_world,
    euler_to_quaternion,
    quaternion_to_euler,
    build_goal_frame,
    project_to_goal_frame,
)
from controllers.lee_controller import (
    LeeControllerParams,
    create_lee_params,
    vel_to_mjx_ctrl,
)
from envs.lidar import compute_lidar_scan
from envs.mjcf_scene import (
    build_scene_xml_with_static_obstacles,
    _STATIC_HFIELD_NAME,
    _STATIC_GEOM_GROUP,
    _DYNAMIC_GEOM_GROUP,
    geomgroup_mask,
    static_obstacle_geom_names,
    dynamic_obstacle_body_names,
    dynamic_obstacle_geom_names,
)
from envs.obstacle_generator import StaticObstacleField, generate_static_obstacles


# ============================================================
# 环境配置
# ============================================================

class EnvConfig(NamedTuple):
    """环境配置常数（不可变）。"""
    max_episode_length: int = 2200
    action_limit: float = 2.0           # m/s
    spawn_distance: float = 24.0        # m
    spawn_edge_inset: float = 0.5       # m，物理 hfield 场景下为边界起飞带保留安全内缩
    height_range_min: float = 0.5       # m
    height_range_max: float = 2.5       # m
    reward_bias: float = 1.0
    penalty_smooth_coeff: float = 0.1
    penalty_height_coeff: float = 8.0
    height_penalty_margin: float = 0.2  # m
    height_min_terminate: float = 0.2   # m
    height_max_terminate: float = 4.0   # m
    reach_goal_dist: float = 0.5        # m
    # ---- 完整环境扩展字段（默认值保持向后兼容）----
    lidar_range: float = 4.0            # m
    n_dynamic_obs: int = 80
    n_nearest_dynamic: int = 5
    collision_radius: float = 0.3       # m
    vel_resample_steps: int = 125       # 每 2s 重采样速度（62.5Hz × 2s）
    sim_dt: float = 0.016               # 仿真步长 (s)，与 MJX 模型一致
    dyn_obs_vel_min: float = 0.5        # m/s
    dyn_obs_vel_max: float = 1.5        # m/s（修复：对应原代码 vel_range=[0.5,1.5]）
    reward_safety_static_coeff: float = 1.0
    reward_safety_dynamic_coeff: float = 1.0
    dyn_obs_local_range: float = 5.0    # m（动态障碍物局部目标采样 x/y 范围，对应原代码 local_range[0]=local_range[1]=5.0）
    dyn_obs_local_range_z: float = 4.5  # m（动态障碍物局部目标采样 z 范围，对应原代码 local_range[2]=4.5）
    dyn_obs_map_range: float = 20.0     # m（动态障碍物 x/y 边界，对应原代码 map_range=[20,20,4.5]）
    # ---- 静态障碍物多样性（对应原代码每次运行重新生成地形）----
    # 0 = 固定静态障碍物（默认，保持向后兼容）
    # N > 0 = 每 N 个训练迭代重新生成静态障碍物（提升训练多样性）
    regenerate_static_obs_interval: int = 0
    physical_static_obstacles: bool = True
    n_static_obs: int = 350
    static_obs_map_range: float = 20.0
    static_scene_margin: float = 1.0
    static_terrain_generator: str = "orbit_discrete"
    static_terrain_hscale: float = 0.1
    static_terrain_vscale: float = 0.1
    static_terrain_platform_width: float = 0.0
    static_obstacle_representation: str = "hfield"
    hfield_cell_size: float = 0.1
    hfield_base_z: float = 0.1
    # ---- LiDAR 扫描范围（对应原代码 mesh_prim_paths 设置）----
    # False = 仅扫描静态障碍物（对应原代码 mesh_prim_paths=["/World/ground"]）
    # True  = 同时扫描静态和动态障碍物
    lidar_scan_dynamic: bool = False
    lidar_use_scene_ray: bool = True
    physical_dynamic_obstacles: bool = True


# ============================================================
# 环境状态（JAX pytree）
# ============================================================

class EnvState(NamedTuple):
    """
    单个环境的完整状态。
    使用 NamedTuple 作为 JAX pytree（可 jit/vmap/tree_map）。
    """
    mjx_data: mjx.Data              # MJX 物理状态
    target_pos: jnp.ndarray         # (3,) 目标位置
    target_dir: jnp.ndarray         # (3,) 起点→目标方向
    height_range: jnp.ndarray       # (2,) [min_h, max_h] 用于高度惩罚
    prev_vel: jnp.ndarray           # (3,) 上一步速度（平滑度惩罚）
    step_count: jnp.ndarray         # () 当前步数
    done: jnp.ndarray               # () 是否结束
    key: jnp.ndarray                # PRNGKey


# ============================================================
# MJCF 路径
# ============================================================

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
_XML_PATH = os.path.join(_ASSETS_DIR, "hummingbird.xml")


# ============================================================
# 辅助函数
# ============================================================

def _sample_edge_position(
    key_side: jnp.ndarray,
    key_pos: jnp.ndarray,
    key_h: jnp.ndarray,
    spawn_dist: float = 24.0,
    edge_inset: float = 0.0,
    h_min: float = 0.5,
    h_max: float = 2.5,
) -> jnp.ndarray:
    """
    从 4 个边缘区域之一采样位置。

    迁移自 env.py:_reset_idx() 和 reset_target()。
    4 个区域: +y边, -y边, +x边, -x边（与原代码 masks/shifts 对应）。

    Args:
        key_side: PRNG key for side selection
        key_pos:  PRNG key for position
        key_h:    PRNG key for height

    Returns:
        (3,) 世界坐标系位置
    """
    # 原代码 masks/shifts:
    # masks = [[1,0,1], [1,0,1], [0,1,1], [0,1,1]]
    # shifts = [[0,24,0], [0,-24,0], [24,0,0], [-24,0,0]]
    masks = jnp.array([
        [1., 0., 1.],
        [1., 0., 1.],
        [0., 1., 1.],
        [0., 1., 1.],
    ])
    edge = max(float(spawn_dist) - float(edge_inset), 0.0)
    shifts = jnp.array([
        [0., edge, 0.],
        [0., -edge, 0.],
        [edge, 0., 0.],
        [-edge, 0., 0.],
    ])

    side_idx = jax.random.randint(key_side, (), 0, 4)
    mask = masks[side_idx]
    shift = shifts[side_idx]

    # 随机位置 ∈ [-24, 24]
    pos_xy = jax.random.uniform(key_pos, (3,), minval=-spawn_dist, maxval=spawn_dist)
    height = jax.random.uniform(key_h, (), minval=h_min, maxval=h_max)
    pos = pos_xy.at[2].set(height)

    return pos * mask + shift


def _extract_root_state(mjx_data: mjx.Data) -> jnp.ndarray:
    """
    从 MJX 数据中提取 13D 状态向量。

    使用 sensordata（与 hummingbird.xml 中传感器定义对应）：
      [0:3]  framepos     → pos
      [3:7]  framequat    → quat [w,x,y,z]
      [7:10] framelinvel  → 世界系线速度
      [10:13] frameangvel → 世界系角速度

    Returns:
        (13,) [pos(3), quat(4), linvel(3), angvel(3)]
    """
    return mjx_data.sensordata[:13]


def _get_obs(
    root_state: jnp.ndarray,
    target_pos: jnp.ndarray,
    target_dir: jnp.ndarray,
) -> jnp.ndarray:
    """
    计算 8D 观测向量。

    迁移自 env.py:_compute_state_and_obs() 第 458-477 行。

    观测量（全在目标坐标系中表达）：
      rpos_clipped_g (3): 指向目标的单位方向
      distance_2d    (1): 水平距离
      distance_z     (1): 垂直距离
      vel_g          (3): 速度

    Args:
        root_state:  (13,) 无人机状态
        target_pos:  (3,) 目标位置
        target_dir:  (3,) 起点→目标方向

    Returns:
        (8,) 观测向量
    """
    pos = root_state[:3]
    vel_w = root_state[7:10]

    rpos = target_pos - pos
    distance = jnp.linalg.norm(rpos) + 1e-6
    distance_2d = jnp.linalg.norm(rpos[:2])
    distance_z = rpos[2]

    # 目标坐标系方向（水平分量）—— 构建一次，投影两次
    target_dir_2d = target_dir.at[2].set(0.0)
    x_axis, y_axis, z_axis = build_goal_frame(target_dir_2d)

    # 单位方向在目标坐标系中的表达
    rpos_unit = rpos / distance
    rpos_clipped_g = project_to_goal_frame(rpos_unit, x_axis, y_axis, z_axis)

    # 速度在目标坐标系中的表达
    vel_g = project_to_goal_frame(vel_w, x_axis, y_axis, z_axis)

    return jnp.concatenate([
        rpos_clipped_g,
        jnp.array([distance_2d]),
        jnp.array([distance_z]),
        vel_g,
    ])  # (8,)


def _compute_reward(
    root_state: jnp.ndarray,
    target_pos: jnp.ndarray,
    prev_vel: jnp.ndarray,
    height_range: jnp.ndarray,
    config: EnvConfig,
) -> jnp.ndarray:
    """
    计算奖励。

    迁移自 env.py:_compute_state_and_obs() 第 548-577 行（无 LiDAR/动态障碍物）。

    奖励组成：
      + reward_vel:     速度在目标方向上的分量（鼓励朝目标飞行）
      + reward_bias:    常数偏置 +1.0（保持奖励为正）
      - penalty_smooth: 速度变化惩罚（鼓励平滑飞行）
      - penalty_height: 高度超界惩罚（限制飞行高度范围）
    """
    pos = root_state[:3]
    vel_w = root_state[7:10]

    rpos = target_pos - pos
    distance = jnp.linalg.norm(rpos) + 1e-6

    # 速度奖励：速度在目标方向上的投影
    vel_direction = rpos / distance
    reward_vel = jnp.sum(vel_w * vel_direction)

    # 平滑度惩罚
    penalty_smooth = jnp.linalg.norm(vel_w - prev_vel)

    # 高度惩罚（二次）
    h = pos[2]
    h_max = height_range[1] + config.height_penalty_margin
    h_min = height_range[0] - config.height_penalty_margin
    penalty_height = jnp.where(
        h > h_max, (h - h_max) ** 2,
        jnp.where(h < h_min, (h_min - h) ** 2, 0.0)
    )

    reward = (reward_vel
              + config.reward_bias
              - penalty_smooth * config.penalty_smooth_coeff
              - penalty_height * config.penalty_height_coeff)
    return reward


def _compute_done(
    root_state: jnp.ndarray,
    step_count: jnp.ndarray,
    config: EnvConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    计算终止条件。

    Returns:
        (terminated, truncated): 两个布尔标量
    """
    h = root_state[2]
    below = h < config.height_min_terminate
    above = h > config.height_max_terminate
    terminated = below | above
    truncated = step_count >= config.max_episode_length
    return terminated, truncated


# ============================================================
# 导航环境类
# ============================================================

class NavigationEnv:
    """
    MJX 导航环境。

    使用模式：
      env = NavigationEnv()
      state, obs = env.reset(key)
      state, obs, reward, done, info = env.step(state, action)

    并行化：
      v_reset = jax.vmap(env.reset)
      v_step = jax.vmap(env.step)
    """

    def __init__(self, config: EnvConfig = EnvConfig()):
        self.config = config
        self.obs_dim = 8
        self.action_dim = 3

        # 加载 MuJoCo 模型（CPU，一次性）
        self.mj_model = mujoco.MjModel.from_xml_path(_XML_PATH)
        self.mjx_model = mjx.put_model(self.mj_model)

        # Lee 控制器参数
        self.ctrl_params = create_lee_params()

        # 模板 mjx_data（用于 reset 中的 .replace()）
        d = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, d)
        d.qpos[:3] = [0.0, 0.0, 2.0]
        d.qpos[3] = 1.0  # 单位四元数 w=1
        mujoco.mj_forward(self.mj_model, d)
        self._template_data = mjx.put_data(self.mj_model, d)

    def reset(self, key: jnp.ndarray) -> Tuple[EnvState, jnp.ndarray]:
        """
        重置单个环境。

        迁移自 env.py:_reset_idx()

        Args:
            key: JAX PRNG key

        Returns:
            (EnvState, obs): 初始状态和 8D 观测
        """
        (key, k_start_side, k_start_pos, k_start_h,
         k_target_side, k_target_pos, k_target_h) = jax.random.split(key, 7)

        # 采样起点和目标
        start_pos = _sample_edge_position(
            k_start_side, k_start_pos, k_start_h,
            self.config.spawn_distance,
            self.config.spawn_edge_inset,
            self.config.height_range_min, self.config.height_range_max,
        )
        target_pos = _sample_edge_position(
            k_target_side, k_target_pos, k_target_h,
            self.config.spawn_distance,
            self.config.spawn_edge_inset,
            self.config.height_range_min, self.config.height_range_max,
        )

        # 目标方向和初始偏航
        target_dir = target_pos - start_pos
        yaw = jnp.arctan2(target_dir[1], target_dir[0])

        # 设置 MJX 状态
        rpy = jnp.array([0.0, 0.0, yaw])
        quat = euler_to_quaternion(rpy)  # (4,) [w,x,y,z]
        qpos = jnp.concatenate([start_pos, quat])  # (7,)
        qvel = jnp.zeros(6)

        # 用模板替换 qpos/qvel，然后 forward 重算传感器
        mjx_data = self._template_data.replace(
            qpos=qpos,
            qvel=qvel,
        )
        mjx_data = mjx.forward(self.mjx_model, mjx_data)

        # 高度范围
        min_h = jnp.minimum(start_pos[2], target_pos[2])
        max_h = jnp.maximum(start_pos[2], target_pos[2])
        height_range = jnp.array([min_h, max_h])

        state = EnvState(
            mjx_data=mjx_data,
            target_pos=target_pos,
            target_dir=target_dir,
            height_range=height_range,
            prev_vel=jnp.zeros(3),
            step_count=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False),
            key=key,
        )

        root_state = _extract_root_state(mjx_data)
        obs = _get_obs(root_state, target_pos, target_dir)
        return state, obs

    def _fast_reset(self, key: jnp.ndarray) -> Tuple[EnvState, jnp.ndarray]:
        """
        快速重置单个环境（跳过 mjx.forward）。

        与 reset() 相同的随机采样和状态构建，但不调用 mjx.forward()。
        初始观测从已知的分析状态直接计算（reset 时 pos、quat、vel 均已知）。

        安全性：下一次 env.step() 中的 mjx.step() 会在积分前执行完整
        前向运动学，确保 MJX 内部状态（xpos, xquat, sensordata 等）
        在首次物理步后恢复一致。

        用于 step_with_autoreset()，避免每步为所有 2048 个环境都执行
        昂贵的 mjx.forward()（其中绝大多数未实际终止）。
        """
        (key, k_start_side, k_start_pos, k_start_h,
         k_target_side, k_target_pos, k_target_h) = jax.random.split(key, 7)

        start_pos = _sample_edge_position(
            k_start_side, k_start_pos, k_start_h,
            self.config.spawn_distance,
            self.config.spawn_edge_inset,
            self.config.height_range_min, self.config.height_range_max,
        )
        target_pos = _sample_edge_position(
            k_target_side, k_target_pos, k_target_h,
            self.config.spawn_distance,
            self.config.spawn_edge_inset,
            self.config.height_range_min, self.config.height_range_max,
        )

        target_dir = target_pos - start_pos
        yaw = jnp.arctan2(target_dir[1], target_dir[0])

        rpy = jnp.array([0.0, 0.0, yaw])
        quat = euler_to_quaternion(rpy)
        qpos = jnp.concatenate([start_pos, quat])
        qvel = jnp.zeros(6)

        # 用模板替换 qpos/qvel，跳过 mjx.forward
        mjx_data = self._template_data.replace(
            qpos=qpos,
            qvel=qvel,
        )

        min_h = jnp.minimum(start_pos[2], target_pos[2])
        max_h = jnp.maximum(start_pos[2], target_pos[2])
        height_range = jnp.array([min_h, max_h])

        state = EnvState(
            mjx_data=mjx_data,
            target_pos=target_pos,
            target_dir=target_dir,
            height_range=height_range,
            prev_vel=jnp.zeros(3),
            step_count=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False),
            key=key,
        )

        # 分析计算初始观测（reset 时状态完全已知）
        root_state = jnp.concatenate([
            start_pos,        # pos (3)
            quat,             # quat (4) [w,x,y,z]
            jnp.zeros(3),     # linvel (3) = 0
            jnp.zeros(3),     # angvel (3) = 0
        ])
        obs = _get_obs(root_state, target_pos, target_dir)
        return state, obs

    def step(
        self,
        state: EnvState,
        action: jnp.ndarray,
    ) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        """
        环境前进一步。

        动作管道:
          1. action (3D, 目标坐标系, [-2,2] m/s) → vec_to_world → 世界系速度
          2. Lee 控制器 → 4 电机推力
          3. MJX 物理推进
          4. 提取状态 → 计算观测/奖励/终止

        Args:
            state:  当前 EnvState
            action: (3,) 目标坐标系中的速度指令

        Returns:
            (new_state, obs, reward, done, info)
        """
        # 当前状态
        root_state = _extract_root_state(state.mjx_data)
        curr_vel = root_state[7:10]

        # 1. 动作变换：目标坐标系 → 世界坐标系
        target_dir_2d = state.target_dir.at[2].set(0.0)
        vel_cmd_world = vec_to_world(action, target_dir_2d)

        # 2. Lee 控制器：速度指令 → MJX ctrl
        ctrl = vel_to_mjx_ctrl(self.ctrl_params, root_state, vel_cmd_world)

        # 3. 物理推进
        mjx_data = state.mjx_data.replace(ctrl=ctrl)
        mjx_data = mjx.step(self.mjx_model, mjx_data)

        # 4. 提取新状态
        new_root_state = _extract_root_state(mjx_data)
        new_step_count = state.step_count + 1

        # 5. 计算观测
        obs = _get_obs(new_root_state, state.target_pos, state.target_dir)

        # 6. 计算奖励
        reward = _compute_reward(
            new_root_state, state.target_pos,
            state.prev_vel, state.height_range,
            self.config,
        )

        # 7. 计算终止
        terminated, truncated = _compute_done(
            new_root_state, new_step_count, self.config,
        )
        done = terminated | truncated

        # reach_goal stat（对应原代码 env.py:583）
        pos = new_root_state[:3]
        rpos_to_target = state.target_pos - pos
        reach_goal = jnp.linalg.norm(rpos_to_target) < self.config.reach_goal_dist

        # 8. 更新状态
        new_state = EnvState(
            mjx_data=mjx_data,
            target_pos=state.target_pos,
            target_dir=state.target_dir,
            height_range=state.height_range,
            prev_vel=new_root_state[7:10],  # 使用本步物理后速度（对应 B: self.prev_drone_vel_w = self.drone.vel_w）
            step_count=new_step_count,
            done=done,
            key=state.key,
        )

        info = {
            "step_count": new_step_count,
            "terminated": terminated,
            "truncated": truncated,
            "reach_goal": reach_goal,
            "collision": jnp.array(False),  # 简化环境无障碍物，collision 恒为 False
        }

        return new_state, obs, reward, done, info


# ============================================================
# 向量化环境封装
# ============================================================

class VectorizedNavigationEnv:
    """
    向量化（并行）环境封装。

    使用 jax.vmap 将单环境 reset/step 并行化到 N 个环境。
    包含自动重置逻辑：done 时自动 reset。
    """

    def __init__(self, num_envs: int, config: EnvConfig = EnvConfig()):
        self.num_envs = num_envs
        self.env = NavigationEnv(config)
        self.config = config
        self.obs_dim = self.env.obs_dim
        self.action_dim = self.env.action_dim

    def reset(self, key: jnp.ndarray) -> Tuple[EnvState, jnp.ndarray]:
        """
        重置所有环境。

        Args:
            key: 单个 PRNG key，内部 split 为 N 个

        Returns:
            (batched_state, batched_obs): 批量状态和观测
        """
        keys = jax.random.split(key, self.num_envs)
        return jax.vmap(self.env.reset)(keys)

    def step(
        self,
        states: EnvState,
        actions: jnp.ndarray,
    ) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        """
        所有环境前进一步。

        Args:
            states:  批量 EnvState（每个字段多一个 leading dim）
            actions: (num_envs, 3) 动作

        Returns:
            (states, obs, rewards, dones, infos)
        """
        return jax.vmap(self.env.step)(states, actions)

    def step_with_autoreset(
        self,
        states: EnvState,
        actions: jnp.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        """
        带自动重置的环境步进。

        done 的环境自动 reset，返回新 episode 的初始观测。
        奖励和 done 仍来自终止前的最后一步。
        """
        new_states, obs, rewards, dones, infos = self.step(states, actions)

        # 为 done 的环境准备 reset（使用快速重置，跳过 mjx.forward）
        reset_keys = jax.random.split(key, self.num_envs)
        reset_states, reset_obs = jax.vmap(self.env._fast_reset)(reset_keys)

        # 条件替换：done → reset, 否则保持
        def _select(reset_val, continue_val):
            if reset_val.ndim <= 1:
                # 标量或 1D 数组（如 step_count(N,), done(N,)）
                # dones(N,) 直接与 (N,) 广播正确
                return jnp.where(dones, reset_val, continue_val)
            else:
                # 多维数组（如 qpos(N,7), target_pos(N,3)）
                # 需要将 dones(N,) reshape 为 (N,1,...) 以正确广播
                shape = [dones.shape[0]] + [1] * (reset_val.ndim - 1)
                return jnp.where(dones.reshape(shape), reset_val, continue_val)

        final_states = jax.tree.map(_select, reset_states, new_states)
        final_obs = jnp.where(dones[:, None], reset_obs, obs)

        return final_states, final_obs, rewards, dones, infos


# ============================================================
# 动态障碍物系统（阶段 3）
# ============================================================

# 类别常量（严格对应原代码 N_w=4, N_h=2）
_N_W = 4
_MAX_OBS_WIDTH      = 1.0   # m，最大障碍物宽度
_MAX_OBS_3D_HEIGHT  = 1.0   # m，3D（长方体）障碍物高度
_MAX_OBS_2D_HEIGHT  = 5.0   # m，2D（圆柱体）障碍物高度
_DYN_OBS_WIDTH_RES  = _MAX_OBS_WIDTH / _N_W   # 0.25m，宽度分辨率

# 每个类别（0-7）的宽度和高度
# 0-3: 长方体，高度=1.0m，宽度 [0.25, 0.50, 0.75, 1.0]
# 4-7: 圆柱体，高度=5.0m，宽度（直径）[0.25, 0.50, 0.75, 1.0]
_CATEGORY_WIDTHS  = jnp.array([0.25, 0.50, 0.75, 1.0,  0.25, 0.50, 0.75, 1.0],  dtype=jnp.float32)
_CATEGORY_HEIGHTS = jnp.array([1.0,  1.0,  1.0,  1.0,  5.0,  5.0,  5.0,  5.0],  dtype=jnp.float32)
_IDENTITY_QUAT = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
_STATIC_GEOMGROUP_MASK = geomgroup_mask(_STATIC_GEOM_GROUP)
_DYNAMIC_GEOMGROUP_MASK = geomgroup_mask(_DYNAMIC_GEOM_GROUP)


def _dynamic_categories_for_count(n_dynamic_obs: int) -> jnp.ndarray:
    """返回固定拓扑动态障碍物类别序列。"""
    if n_dynamic_obs % 8 != 0:
        raise ValueError(
            f"n_dynamic_obs ({n_dynamic_obs}) must be divisible by 8 "
            f"(4 width categories × 2 height categories)."
        )
    return jnp.repeat(jnp.arange(8, dtype=jnp.int32), n_dynamic_obs // 8)


def _dynamic_half_extents_from_categories(categories: jnp.ndarray) -> jnp.ndarray:
    """根据类别返回动态障碍物物理几何半尺寸。"""
    widths = _CATEGORY_WIDTHS[categories]
    heights = _CATEGORY_HEIGHTS[categories]
    return jnp.stack([widths / 2.0, widths / 2.0, heights / 2.0], axis=-1)


def _dynamic_geom_types_from_categories(categories: jnp.ndarray) -> tuple[str, ...]:
    """根据类别返回动态障碍物 MuJoCo geom 类型。"""
    return tuple("box" if int(cat) < 4 else "cylinder" for cat in np.asarray(categories))


def _apply_dynamic_obstacles_to_data(
    mjx_data: mjx.Data,
    dyn_obs: "DynObsState",
    dynamic_mocap_ids: jnp.ndarray,
) -> mjx.Data:
    """将动态障碍物位姿写入 MJX mocap 数据。"""
    if dynamic_mocap_ids.size == 0:
        return mjx_data

    mocap_pos = mjx_data.mocap_pos.at[dynamic_mocap_ids].set(dyn_obs.positions)
    mocap_quat = mjx_data.mocap_quat.at[dynamic_mocap_ids].set(
        jnp.broadcast_to(_IDENTITY_QUAT, (dynamic_mocap_ids.shape[0], 4))
    )
    return mjx_data.replace(mocap_pos=mocap_pos, mocap_quat=mocap_quat)


class DynObsState(NamedTuple):
    """
    动态障碍物完整状态（存入 FullEnvState）。

    所有字段 shape[0] = N_dyn。
    """
    positions:      jnp.ndarray  # (N_dyn, 3) 世界坐标中心
    velocities:     jnp.ndarray  # (N_dyn, 3) 当前速度 (m/s)
    goals:          jnp.ndarray  # (N_dyn, 3) 当前目标位置
    categories:     jnp.ndarray  # (N_dyn,) int32，类别 0~7
    origins:        jnp.ndarray  # (N_dyn, 3) 初始位置（用于局部目标采样）
    dyn_step_count: jnp.ndarray  # () 全局步计数器（跨 episode 不重置，对应 B: dyn_obs_step_count）


class FullEnvState(NamedTuple):
    """
    完整环境状态，包含动态障碍物。

    与 EnvState 相比新增 dyn_obs 字段。
    FullNavigationEnv 使用此状态；NavigationEnv 仍使用 EnvState。
    """
    mjx_data:     mjx.Data
    target_pos:   jnp.ndarray   # (3,)
    target_dir:   jnp.ndarray   # (3,)
    height_range: jnp.ndarray   # (2,)
    prev_vel:     jnp.ndarray   # (3,)
    step_count:   jnp.ndarray   # ()
    done:         jnp.ndarray   # ()
    key:          jnp.ndarray   # PRNGKey
    dyn_obs:      DynObsState


class ObsBundle(NamedTuple):
    """
    完整多模态观测（4 分量）。

    对应原代码 observation_spec 中的 4 个键：
      state(8D), lidar(1×36×4), direction(3D), dynamic_obstacle(1×5×10)
    注：此处 dynamic_obs 形状为 (5, 10)（去掉原代码的批次维度 1）。
    """
    state:       jnp.ndarray  # (8,)
    lidar:       jnp.ndarray  # (1, 36, 4)
    direction:   jnp.ndarray  # (3,) target_dir_2d 归一化
    dynamic_obs: jnp.ndarray  # (5, 10)


def _prepare_lidar_scene_data(mjx_model: mjx.Model, mjx_data: mjx.Data) -> mjx.Data:
    """
    为 MJX scene ray 准备最小必需的几何状态。

    `mjx.ray()` 只依赖当前位置/姿态相关的几何量；在 autoreset 快路径中
    无需执行完整 `mjx.forward()`，只做 `kinematics()` 即可保持 LiDAR 与当前
    静态场景一致。
    """
    return mjx.kinematics(mjx_model, mjx_data)


def _has_contact_with_geom_set(
    mjx_data: mjx.Data,
    collision_hull_geom_id: int,
    geom_ids: jnp.ndarray,
) -> jnp.ndarray:
    """判断无人机碰撞代理是否与给定 geom 集合发生接触。"""
    if geom_ids.size == 0:
        return jnp.array(False)

    data_impl = getattr(mjx_data, "_impl", mjx_data)
    ncon = jnp.asarray(data_impl.ncon, dtype=jnp.int32)
    contact_geom = data_impl.contact.geom
    contact_dist = data_impl.contact.dist
    valid = jnp.arange(contact_geom.shape[0], dtype=jnp.int32) < ncon
    hits_hull = jnp.any(contact_geom == collision_hull_geom_id, axis=-1)
    hits_target = jnp.any(contact_geom[:, :, None] == geom_ids[None, None, :], axis=(1, 2))
    penetrating = contact_dist <= 1e-6
    return jnp.any(valid & hits_hull & hits_target & penetrating)


def _has_static_contact(
    mjx_data: mjx.Data,
    collision_hull_geom_id: int,
    static_geom_ids: jnp.ndarray,
) -> jnp.ndarray:
    """精确判断无人机碰撞代理与静态场景 geom 是否发生接触。"""
    return _has_contact_with_geom_set(mjx_data, collision_hull_geom_id, static_geom_ids)


def _has_dynamic_contact(
    mjx_data: mjx.Data,
    collision_hull_geom_id: int,
    dynamic_geom_ids: jnp.ndarray,
) -> jnp.ndarray:
    """精确判断无人机碰撞代理与动态障碍物 geom 是否发生接触。"""
    return _has_contact_with_geom_set(mjx_data, collision_hull_geom_id, dynamic_geom_ids)


def _reset_dynamic_obstacles(
    key: jnp.ndarray,
    config: EnvConfig,
) -> DynObsState:
    """
    初始化动态障碍物状态。

    80 个障碍物，8 类均匀分布（每类 10 个）：
      类别 0-3: 长方体（高=1m），随机 z 中心
      类别 4-7: 圆柱体（高=5m），z 中心固定在 2.5m

    修复（对应原代码 env.py:172-204）：
      - 使用拒绝采样保证障碍物间最小间距（对应 B 的 check_pos_validity）
      - 记录 origins（初始位置），供后续局部目标采样使用。
      - 初始化 dyn_step_count=0（全局步计数器）。
    """
    n_dyn = config.n_dynamic_obs
    n_per_cat = n_dyn // 8
    # 与原代码保持一致：障碍物数量必须能被 8 整除（4 宽度类别 × 2 高度类别）
    assert n_dyn % 8 == 0, (
        f"n_dynamic_obs ({n_dyn}) must be divisible by 8 "
        f"(4 width categories × 2 height categories). "
        f"Suggested values: 8, 16, 32, 40, 48, 56, 64, 72, 80, 88, 96, ..."
    )

    key, pos_key, goal_key, vel_key = jax.random.split(key, 4)

    categories = _dynamic_categories_for_count(n_dyn)  # (N_dyn,)
    heights = _CATEGORY_HEIGHTS[categories]  # (N_dyn,)
    is_3d = categories < 4

    # 初始位置（网格抖动法，JAX 兼容，保证障碍物间最小间距）
    # 对应 B: check_pos_validity 的功能目标
    # 将地图分成 grid_n×grid_n 网格，每格放一个障碍物并加随机抖动
    map_range = config.dyn_obs_map_range
    map_size = 2.0 * map_range
    grid_n = int(math.ceil(math.sqrt(n_dyn)))  # 网格边长
    cell_size = map_size / grid_n

    pk1, pk2, pk3 = jax.random.split(pos_key, 3)

    # 随机选取 n_dyn 个网格单元（洗牌后取前 n_dyn 个）
    all_indices = jnp.arange(grid_n * grid_n)
    shuffled = jax.random.permutation(pk1, all_indices)[:n_dyn]
    grid_x = (shuffled % grid_n).astype(jnp.float32)
    grid_y = (shuffled // grid_n).astype(jnp.float32)

    # 网格中心 + 随机抖动（±0.3*cell_size），保证最小间距 ≈ 0.4*cell_size
    cx = -map_range + (grid_x + 0.5) * cell_size
    cy = -map_range + (grid_y + 0.5) * cell_size
    jitter_range = cell_size * 0.3
    jitter = jax.random.uniform(pk2, (n_dyn, 2), minval=-jitter_range, maxval=jitter_range)
    pos_x = cx + jitter[:, 0]
    pos_y = cy + jitter[:, 1]

    # Z 坐标：3D 障碍物随机 [0, 4.5]，地面障碍物固定 height/2
    z_3d = jax.random.uniform(pk3, (n_dyn,), minval=0.0, maxval=4.5)
    z_ground = heights / 2.0
    pos_z = jnp.where(is_3d, z_3d, z_ground)

    positions = jnp.stack([pos_x, pos_y, pos_z], axis=-1)  # (N_dyn, 3)

    # origins = 初始位置（用于局部目标采样，对应原代码 dyn_obs_origin）
    origins = positions

    # 初始目标：在 origins 局部范围内采样（对应原代码 local_range=[5.0, 5.0, 4.5]）
    # x/y 使用 dyn_obs_local_range=5.0，z 使用 dyn_obs_local_range_z=4.5
    key, goal_xy_key, goal_z_key = jax.random.split(goal_key, 3)
    goal_offset_xy = jax.random.uniform(
        goal_xy_key, (n_dyn, 2),
        minval=-config.dyn_obs_local_range, maxval=config.dyn_obs_local_range,
    )
    goal_offset_z = jax.random.uniform(
        goal_z_key, (n_dyn, 1),
        minval=-config.dyn_obs_local_range_z, maxval=config.dyn_obs_local_range_z,
    )
    goal_offset = jnp.concatenate([goal_offset_xy, goal_offset_z], axis=-1)
    raw_goals = origins + goal_offset
    # 裁剪到地图范围（xy: dyn_obs_map_range=20m，z: [0.0, 4.5]，对应原代码 map_range=[20,20,4.5] clamp min=0）
    goal_xy = jnp.clip(raw_goals[:, :2], -config.dyn_obs_map_range, config.dyn_obs_map_range)
    goal_z_3d = jnp.clip(raw_goals[:, 2], 0.0, config.dyn_obs_local_range_z)
    goal_z = jnp.where(is_3d, goal_z_3d, heights / 2.0)
    goals = jnp.concatenate([goal_xy, goal_z[:, None]], axis=-1)  # (N_dyn, 3)

    # 初始速度：朝向目标，大小随机
    diff = goals - positions
    dist = jnp.linalg.norm(diff, axis=-1, keepdims=True).clip(1e-6)
    vel_dir = diff / dist
    vel_mag = jax.random.uniform(
        vel_key, (n_dyn, 1),
        minval=config.dyn_obs_vel_min, maxval=config.dyn_obs_vel_max,
    )
    velocities = vel_dir * vel_mag  # (N_dyn, 3)

    return DynObsState(
        positions=positions,
        velocities=velocities,
        goals=goals,
        categories=categories,
        origins=origins,
        dyn_step_count=jnp.array(0, dtype=jnp.int32),
    )


def _step_dynamic_obstacles(
    dyn_obs: DynObsState,
    step_count: jnp.ndarray,
    key: jnp.ndarray,
    config: EnvConfig,
) -> DynObsState:
    """
    动态障碍物单步更新。

    对应原代码 move_dyn_obs()：
      1. 位置积分：pos += vel * dt
      2. 达到目标（dist < 0.5m）→ 在 origins 局部范围内采样新目标 + 重采样速度
      3. 全局定时重采样（每 vel_resample_steps 步）→ 重采样速度

    使用 jnp.where 代替所有条件分支，保证 JIT/vmap 安全。

    注：速度重采样使用 dyn_step_count（全局计数器，跨 episode 不重置），
    对应 B 的 self.dyn_obs_step_count，而非 episode 的 step_count。
    """
    key, goal_key, vel_key = jax.random.split(key, 3)
    n_dyn = dyn_obs.positions.shape[0]

    heights = _CATEGORY_HEIGHTS[dyn_obs.categories]   # (N_dyn,)
    is_3d = dyn_obs.categories < 4

    # ---- 位置更新 ----
    new_positions = dyn_obs.positions + dyn_obs.velocities * config.sim_dt

    # 边界裁剪（使用 dyn_obs_map_range=20m，对应原代码 map_range=[20,20,4.5]）
    new_positions = jnp.clip(
        new_positions,
        jnp.array([-config.dyn_obs_map_range, -config.dyn_obs_map_range, 0.1]),
        jnp.array([ config.dyn_obs_map_range,  config.dyn_obs_map_range, 6.0]),
    )

    # ---- 达到目标检测 ----
    diff_to_goal = dyn_obs.goals - new_positions
    dist_to_goal = jnp.linalg.norm(diff_to_goal, axis=-1)  # (N_dyn,)
    at_goal = dist_to_goal < 0.5                            # (N_dyn,)

    # ---- 新目标采样（在 origins 局部范围内，对应原代码 local_range=[5.0,5.0,4.5]）----
    # x/y 使用 dyn_obs_local_range=5.0，z 使用 dyn_obs_local_range_z=4.5
    key, goal_xy_key, goal_z_key = jax.random.split(goal_key, 3)
    goal_offset_xy = jax.random.uniform(
        goal_xy_key, (n_dyn, 2),
        minval=-config.dyn_obs_local_range, maxval=config.dyn_obs_local_range,
    )
    goal_offset_z = jax.random.uniform(
        goal_z_key, (n_dyn, 1),
        minval=-config.dyn_obs_local_range_z, maxval=config.dyn_obs_local_range_z,
    )
    goal_offset = jnp.concatenate([goal_offset_xy, goal_offset_z], axis=-1)
    raw_goals = dyn_obs.origins + goal_offset
    new_goal_xy = jnp.clip(raw_goals[:, :2], -config.dyn_obs_map_range, config.dyn_obs_map_range)
    goal_z_3d = jnp.clip(raw_goals[:, 2], 0.0, config.dyn_obs_local_range_z)  # 对应原代码 clamp min=0
    new_goal_z = jnp.where(is_3d, goal_z_3d, heights / 2.0)
    new_goals_sampled = jnp.concatenate([new_goal_xy, new_goal_z[:, None]], axis=-1)

    # 更新目标（只替换到达目标的障碍物）
    new_goals = jnp.where(at_goal[:, None], new_goals_sampled, dyn_obs.goals)

    # ---- 全局定时速度重采样（使用 dyn_step_count，对应 B: self.dyn_obs_step_count）----
    new_dyn_step_count = dyn_obs.dyn_step_count + 1
    global_resample = (new_dyn_step_count % config.vel_resample_steps) == 0  # ()

    # 需要重采样的障碍物：到达目标 OR 全局定时
    do_resample = at_goal | global_resample   # (N_dyn,)（标量广播到 N_dyn）

    # 新速度：朝向更新后的目标
    diff_new = new_goals - new_positions
    dist_new = jnp.linalg.norm(diff_new, axis=-1, keepdims=True).clip(1e-6)
    new_vel_dir = diff_new / dist_new
    new_vel_mag = jax.random.uniform(
        vel_key, (n_dyn, 1),
        minval=config.dyn_obs_vel_min, maxval=config.dyn_obs_vel_max,
    )
    new_velocities = new_vel_dir * new_vel_mag

    # 条件更新速度
    velocities = jnp.where(do_resample[:, None], new_velocities, dyn_obs.velocities)

    return DynObsState(
        positions=new_positions,
        velocities=velocities,
        goals=new_goals,
        categories=dyn_obs.categories,
        origins=dyn_obs.origins,   # origins 不变
        dyn_step_count=new_dyn_step_count,
    )


# ============================================================
# 完整观测计算（阶段 4）
# ============================================================

def _get_dynamic_obs_features(
    drone_pos: jnp.ndarray,   # (3,)
    target_dir: jnp.ndarray,  # (3,)
    dyn_obs: DynObsState,
    n_nearest: int = 5,
    lidar_range: float = 4.0,
    goal_frame: tuple = None,
) -> jnp.ndarray:
    """
    提取最近 n_nearest 个动态障碍物的 10D 特征向量。

    对应原代码（env.py 第 488-518 行）：
      [rpos_gn(3), dist_2d(1), dist_z(1), vel_g(3), width_cat(1), height_cat(1)]

    所有计算在目标坐标系（goal frame）中表达。

    修复（对应原代码 env.py:486, 489-514）：
      1. 圆柱体（categories >= 4）障碍物的相对位置 z 分量归零（原代码 line 486）
      2. 超出 lidar_range 的障碍物特征归零（原代码 lines 489-514 range mask）

    Args:
        drone_pos:   (3,) 无人机世界坐标
        target_dir:  (3,) 目标方向
        dyn_obs:     DynObsState（N_dyn 个障碍物）
        n_nearest:   选取最近的 k 个
        lidar_range: 感知范围（超出此距离的障碍物特征归零）
        goal_frame:  预计算的 (x_axis, y_axis, z_axis)，为 None 时自动构建

    Returns:
        (n_nearest, 10) 动态障碍物特征
    """
    if goal_frame is not None:
        x_axis, y_axis, z_axis = goal_frame
    else:
        target_dir_2d = target_dir.at[2].set(0.0)
        x_axis, y_axis, z_axis = build_goal_frame(target_dir_2d)
    widths  = _CATEGORY_WIDTHS[dyn_obs.categories]   # (N_dyn,)
    heights = _CATEGORY_HEIGHTS[dyn_obs.categories]  # (N_dyn,)
    is_2d   = dyn_obs.categories >= 4                # (N_dyn,) 圆柱体标志

    # 相对位置（世界坐标系）
    rpos = dyn_obs.positions - drone_pos[None, :]  # (N_dyn, 3)

    # Bug 1.2 修复: 圆柱体障碍物的 z 分量归零（对应原代码 env.py:486）
    rpos_2d_zero = rpos.at[:, 2].set(jnp.where(is_2d, 0.0, rpos[:, 2]))  # (N_dyn, 3)

    # 按 2D 距离选取最近 k 个
    dist_2d_all = jnp.linalg.norm(rpos[:, :2], axis=-1)  # (N_dyn,)
    _, idx = jax.lax.top_k(-dist_2d_all, n_nearest)      # 用负值取最小

    closest_rpos    = rpos_2d_zero[idx]                   # (k, 3)，圆柱体z已归零
    closest_vel     = dyn_obs.velocities[idx]             # (k, 3)
    closest_widths  = widths[idx]                         # (k,)
    closest_heights = heights[idx]                        # (k,)
    closest_dist_2d = dist_2d_all[idx]                    # (k,) 2D 距离（用于 range mask）

    # 3D 距离（分母安全）
    dist_3d = jnp.linalg.norm(closest_rpos, axis=-1, keepdims=True).clip(1e-6)  # (k, 1)

    # 目标坐标系中的相对位置、速度（使用预计算的坐标轴，避免重复构建 frame）
    rpos_g  = jax.vmap(lambda r: project_to_goal_frame(r, x_axis, y_axis, z_axis))(closest_rpos)   # (k, 3)
    rpos_gn = rpos_g / dist_3d                                                        # (k, 3) 归一化

    # 2D 水平距离 / Z 距离（均在 goal frame 中）
    dist_2d = jnp.linalg.norm(rpos_g[:, :2], axis=-1, keepdims=True)  # (k, 1)
    dist_z  = rpos_g[:, 2:3]                                           # (k, 1)

    # 速度在目标坐标系中的表达
    vel_g = jax.vmap(lambda v: project_to_goal_frame(v, x_axis, y_axis, z_axis))(closest_vel)  # (k, 3)

    # 宽度类别: width / width_res - 1 ∈ [0, 1, 2, 3]
    width_cat  = (closest_widths / _DYN_OBS_WIDTH_RES - 1.0)[:, None]  # (k, 1)

    # 高度类别：3D (height=1.0) → 1.0，2D (height=5.0) → 0.0
    # 注意：此编码与原代码 env.py:513 一致（2D障碍物=0，3D障碍物=1.0），
    # 看似反直觉（0=2D而非0=3D），但为保持与B的行为一致，此处刻意保留。
    height_cat = jnp.where(
        closest_heights[:, None] > _MAX_OBS_3D_HEIGHT,
        jnp.zeros_like(closest_heights[:, None]),
        closest_heights[:, None],
    )  # (k, 1)

    features = jnp.concatenate(
        [rpos_gn, dist_2d, dist_z, vel_g, width_cat, height_cat], axis=-1
    )  # (k, 10)

    # Bug 1.3 修复: 超出 lidar_range 的障碍物特征归零（对应原代码 env.py:489-514）
    in_range = (closest_dist_2d <= lidar_range)[:, None]  # (k, 1) bool
    return features * in_range  # (k, 10)


def _get_obs_full(
    root_state: jnp.ndarray,    # (13,)
    target_pos: jnp.ndarray,    # (3,)
    target_dir: jnp.ndarray,    # (3,)
    dyn_obs: DynObsState,
    static_positions: jnp.ndarray,     # (N_s, 3)
    static_half_extents: jnp.ndarray,  # (N_s, 3)
    config: EnvConfig,
    mjx_model: mjx.Model | None = None,
    mjx_data: mjx.Data | None = None,
    drone_body_id: int = -1,
    scene_static_geomgroup: tuple[int, ...] = (),
    scene_dynamic_geomgroup: tuple[int, ...] = (),
    scene_dynamic_geoms: bool = False,
) -> ObsBundle:
    """
    计算完整 4 分量多模态观测。

    对应原代码 _compute_state_and_obs() 的完整版本。

    Args:
        root_state:           (13,) [pos, quat, linvel, angvel]
        target_pos:           (3,)
        target_dir:           (3,)
        dyn_obs:              DynObsState
        static_positions:     (N_s, 3) 静态障碍物中心
        static_half_extents:  (N_s, 3) 静态障碍物半尺寸
        config:               EnvConfig

    Returns:
        ObsBundle
    """
    pos  = root_state[:3]
    quat = root_state[3:7]  # [w, x, y, z]

    # 1. 基础 8D 状态观测（与简化环境相同）
    state = _get_obs(root_state, target_pos, target_dir)

    # 2. 目标方向（归一化 target_dir_2d，3D）—— 构建一次 goal frame，复用于动态障碍物特征
    target_dir_2d = target_dir.at[2].set(0.0)
    x_axis, y_axis, z_axis = build_goal_frame(target_dir_2d)
    eps = 1e-7
    direction = target_dir_2d / (jnp.linalg.norm(target_dir_2d) + eps)  # (3,)

    # 3. 从四元数提取偏航角（yaw-only 旋转）
    w, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    yaw = jnp.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy ** 2 + qz ** 2))

    # 4. 动态障碍物的 AABB 半尺寸（用于 LiDAR）
    dyn_widths  = _CATEGORY_WIDTHS[dyn_obs.categories]   # (N_dyn,)
    dyn_heights = _CATEGORY_HEIGHTS[dyn_obs.categories]  # (N_dyn,)
    dyn_half_extents = jnp.stack(
        [dyn_widths / 2.0, dyn_widths / 2.0, dyn_heights / 2.0], axis=-1
    )  # (N_dyn, 3)

    scene_ray_enabled = (
        config.lidar_use_scene_ray and (
            config.physical_static_obstacles or
            (scene_dynamic_geoms and config.lidar_scan_dynamic)
        )
    )

    # 5. LiDAR 扫描
    lidar = compute_lidar_scan(
        pos, yaw,
        static_positions, static_half_extents,
        dyn_obs.positions, dyn_half_extents,
        config.lidar_range,
        scan_dynamic=config.lidar_scan_dynamic,
        mjx_model=mjx_model if scene_ray_enabled else None,
        mjx_data=mjx_data if scene_ray_enabled else None,
        bodyexclude=drone_body_id,
        scene_static_geomgroup=scene_static_geomgroup,
        scene_dynamic_geomgroup=scene_dynamic_geomgroup,
        scene_dynamic_geoms=scene_dynamic_geoms,
    )  # (1, 36, 4)

    # 6. 最近 n 个动态障碍物特征（复用 goal frame）
    dynamic_obs = _get_dynamic_obs_features(
        pos, target_dir, dyn_obs, config.n_nearest_dynamic, config.lidar_range,
        goal_frame=(x_axis, y_axis, z_axis),
    )  # (5, 10)

    return ObsBundle(
        state=state,
        lidar=lidar,
        direction=direction,
        dynamic_obs=dynamic_obs,
    )


def _check_dynamic_collision(
    drone_pos: jnp.ndarray,  # (3,)
    dyn_obs: DynObsState,
    config: EnvConfig,
) -> jnp.ndarray:
    """
    检测无人机与动态障碍物的碰撞。

    对应原代码（env.py 第 521-528 行）：
      2D collision: dist_2d ≤ (width/2 + collision_radius)
      Z  collision: |dz|   ≤ (height/2 + collision_radius)
      碰撞 = 两条件同时成立

    只检查最近 n_nearest 个障碍物（与原代码保持一致）。

    修复（对应原代码 env.py:522）：
      超出 lidar_range 的障碍物即使满足距离条件也不触发碰撞。

    Returns:
        () bool，True = 发生碰撞
    """
    n_nearest = config.n_nearest_dynamic
    widths  = _CATEGORY_WIDTHS[dyn_obs.categories]   # (N_dyn,)
    heights = _CATEGORY_HEIGHTS[dyn_obs.categories]  # (N_dyn,)

    rpos = dyn_obs.positions - drone_pos[None, :]  # (N_dyn, 3)

    # 按 2D 距离选 k 近
    dist_2d_all = jnp.linalg.norm(rpos[:, :2], axis=-1)
    _, idx = jax.lax.top_k(-dist_2d_all, n_nearest)

    closest_rpos    = rpos[idx]       # (k, 3)
    closest_widths  = widths[idx]     # (k,)
    closest_heights = heights[idx]    # (k,)

    dist_2d = jnp.linalg.norm(closest_rpos[:, :2], axis=-1)   # (k,)
    dist_z  = jnp.abs(closest_rpos[:, 2])                      # (k,)

    col_2d = dist_2d <= (closest_widths  / 2.0 + config.collision_radius)
    col_z  = dist_z  <= (closest_heights / 2.0 + config.collision_radius)

    # Bug 1.4 修复: 超出 lidar_range 的障碍物不参与碰撞检测（对应原代码 env.py:522）
    in_range = dist_2d <= config.lidar_range  # (k,)

    return jnp.any(col_2d & col_z & in_range)


def _compute_done_full(
    root_state: jnp.ndarray,
    step_count: jnp.ndarray,
    lidar_scan: jnp.ndarray,   # (1, 36, 4)
    dyn_obs: DynObsState,
    config: EnvConfig,
    static_contact: jnp.ndarray = None,
    dynamic_contact: jnp.ndarray = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    完整终止条件（静态碰撞 + 动态碰撞 + 高度越界 + 超步数）。

    Returns:
        (terminated, truncated)
    """
    # 基础终止（高度越界）
    terminated_basic, truncated = _compute_done(root_state, step_count, config)

    # 静态碰撞：优先使用 MuJoCo 真实接触；若未启用真实物理障碍物，则退回 LiDAR 近似。
    if static_contact is None:
        static_collision = jnp.max(lidar_scan) > (config.lidar_range - config.collision_radius)
    else:
        static_collision = static_contact

    # 动态碰撞
    pos = root_state[:3]
    if dynamic_contact is None:
        dynamic_collision = _check_dynamic_collision(pos, dyn_obs, config)
    else:
        dynamic_collision = dynamic_contact

    terminated = terminated_basic | static_collision | dynamic_collision
    return terminated, truncated


# ============================================================
# 完整奖励函数（阶段 5）
# ============================================================

def _compute_reward_full(
    root_state: jnp.ndarray,     # (13,)
    target_pos: jnp.ndarray,     # (3,)
    prev_vel: jnp.ndarray,       # (3,)
    height_range: jnp.ndarray,   # (2,)
    lidar_scan: jnp.ndarray,     # (1, 36, 4)
    dyn_obs: DynObsState,
    config: EnvConfig,
) -> jnp.ndarray:
    """
    完整奖励函数（包含静态/动态安全奖励）。

    严格对应原代码（env.py 第 548-577 行）：
      total = reward_vel + 1.0
            + log(clamp(lidar_range - lidar_scan, 1e-6)).mean()
                * reward_safety_static_coeff
            + log(clamp(dist_dyn - width/2, 1e-6, lidar_range)).mean()
                * reward_safety_dynamic_coeff
            - penalty_smooth * 0.1
            - penalty_height * 8.0
    """
    pos    = root_state[:3]
    vel_w  = root_state[7:10]

    # 基础奖励（速度 + 常数偏置 + 平滑 + 高度惩罚）
    rpos     = target_pos - pos
    distance = jnp.linalg.norm(rpos) + 1e-6
    vel_dir  = rpos / distance
    reward_vel = jnp.sum(vel_w * vel_dir)

    penalty_smooth = jnp.linalg.norm(vel_w - prev_vel)
    h     = pos[2]
    h_max = height_range[1] + config.height_penalty_margin
    h_min = height_range[0] - config.height_penalty_margin
    penalty_height = jnp.where(
        h > h_max, (h - h_max) ** 2,
        jnp.where(h < h_min, (h_min - h) ** 2, 0.0),
    )

    # 静态安全奖励：log(distance_to_obstacle).mean()
    # lidar_scan = lidar_range - distance → distance = lidar_range - lidar_scan
    dist_static = config.lidar_range - lidar_scan.reshape(-1)  # (144,)
    reward_safety_static = jnp.mean(
        jnp.log(jnp.clip(dist_static, 1e-6, config.lidar_range))
    )

    # 动态安全奖励：仅使用最近 n_nearest 个障碍物（对应原代码 env.py:529-533）
    # B: closest_dyn_obs_distance_reward = rpos.norm - width/2
    #    out-of-range → lidar_range；然后 log(...).mean() over n_nearest
    widths = _CATEGORY_WIDTHS[dyn_obs.categories]  # (N_dyn,)
    rpos_dyn = dyn_obs.positions - pos[None, :]    # (N_dyn, 3)
    
    # 【修复核心】将圆柱体（categories >= 4）的 Z 轴相对距离强行置 0
    is_2d = dyn_obs.categories >= 4
    rpos_dyn = rpos_dyn.at[:, 2].set(jnp.where(is_2d, 0.0, rpos_dyn[:, 2]))
    
    # 随后再进行距离排序和 3D 距离计算
    dist_2d_dyn = jnp.linalg.norm(rpos_dyn[:, :2], axis=-1)  # (N_dyn,)
    _, idx = jax.lax.top_k(-dist_2d_dyn, config.n_nearest_dynamic)
    
    closest_rpos   = rpos_dyn[idx]       # (n, 3)  此时圆柱体的 z 已经是 0 了
    closest_widths = widths[idx]         # (n,)
    
    # 这里的 3D 距离对于圆柱体实际上就是 2D 距离，物理逻辑自洽
    dist_3d = jnp.linalg.norm(closest_rpos, axis=-1)  
    dist_surface = dist_3d - closest_widths / 2.0

    # 按 2D 距离选最近 n_nearest 个障碍物
    _, idx = jax.lax.top_k(-dist_2d_dyn, config.n_nearest_dynamic)
    closest_rpos   = rpos_dyn[idx]       # (n, 3)
    closest_widths = widths[idx]         # (n,)
    closest_dist_2d = dist_2d_dyn[idx]   # (n,) 2D 距离（用于 range mask）

    # 3D 距离到障碍物表面（对应原代码 rpos.norm(dim=-1) - width/2）
    dist_3d = jnp.linalg.norm(closest_rpos, axis=-1)  # (n,)
    dist_surface = dist_3d - closest_widths / 2.0      # (n,)

    # 超出 lidar_range 的障碍物设为 lidar_range（对应原代码 range_mask 赋值）
    in_range = closest_dist_2d <= config.lidar_range
    dist_surface_masked = jnp.where(in_range, dist_surface, config.lidar_range)

    reward_safety_dynamic = jnp.mean(
        jnp.log(jnp.clip(dist_surface_masked, 1e-6, config.lidar_range))
    )

    total = (
        reward_vel
        + config.reward_bias
        + reward_safety_static  * config.reward_safety_static_coeff
        + reward_safety_dynamic * config.reward_safety_dynamic_coeff
        - penalty_smooth        * config.penalty_smooth_coeff
        - penalty_height        * config.penalty_height_coeff
    )
    return total


# ============================================================
# 完整导航环境类
# ============================================================

class FullNavigationEnv:
    """
    带静态/动态障碍物、LiDAR 传感器的完整导航环境。

    观测空间（多模态 ObsBundle）：
      state(8D), lidar(1×36×4), direction(3D), dynamic_obs(5×10)

    使用模式：
      env = FullNavigationEnv()
      state, obs = env.reset(key)
      state, obs, reward, done, info = env.step(state, action)

    并行化：
      v_reset = jax.vmap(env.reset)
      v_step  = jax.vmap(env.step)
    """

    def __init__(
        self,
        config: EnvConfig = EnvConfig(),
        static_obs_seed: int = 42,
    ):
        self.config = config
        self.obs_dim = 8        # state 分量维度（完整观测见 ObsBundle）
        self.action_dim = 3

        # Lee 控制器
        self.ctrl_params = create_lee_params()
        self.dynamic_categories = _dynamic_categories_for_count(self.config.n_dynamic_obs)
        self.dynamic_half_extents = _dynamic_half_extents_from_categories(self.dynamic_categories)
        self.dynamic_geom_types = _dynamic_geom_types_from_categories(self.dynamic_categories)
        self.dynamic_body_names = dynamic_obstacle_body_names(self.config.n_dynamic_obs)
        self.dynamic_geom_names = dynamic_obstacle_geom_names(self.config.n_dynamic_obs)

        # 生成静态障碍物场（训练开始时调用一次）
        seed_key = jax.random.PRNGKey(static_obs_seed)
        self.static_field = generate_static_obstacles(
            seed_key,
            n_obstacles=self.config.n_static_obs,
            area_size=self.config.static_obs_map_range,
            generator_mode=self.config.static_terrain_generator,
            horizontal_scale=self.config.static_terrain_hscale,
            vertical_scale=self.config.static_terrain_vscale,
            platform_width=self.config.static_terrain_platform_width,
        )
        self.scene_version = -1
        self._rebuild_physics_scene()

    def _rebuild_physics_scene(self) -> None:
        """根据当前静态障碍物场重建 MuJoCo/MJX 场景。"""
        scene_half_extent = max(self.config.spawn_distance, self.config.static_obs_map_range) + self.config.static_scene_margin
        if self.config.physical_static_obstacles:
            scene_xml = build_scene_xml_with_static_obstacles(
                _XML_PATH,
                self.static_field,
                area_size=scene_half_extent,
                terrain_representation=self.config.static_obstacle_representation,
                hfield_cell_size=self.config.hfield_cell_size,
                hfield_base_z=self.config.hfield_base_z,
                dynamic_positions=np.zeros((self.config.n_dynamic_obs, 3), dtype=np.float32) if self.config.physical_dynamic_obstacles else None,
                dynamic_half_extents=np.asarray(self.dynamic_half_extents) if self.config.physical_dynamic_obstacles else None,
                dynamic_geom_types=self.dynamic_geom_types if self.config.physical_dynamic_obstacles else None,
            )
            self.mj_model = mujoco.MjModel.from_xml_string(scene_xml)
        else:
            self.mj_model = mujoco.MjModel.from_xml_path(_XML_PATH)

        self.mjx_model = mjx.put_model(self.mj_model)

        d = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, d)
        d.qpos[:3] = [0.0, 0.0, 2.0]
        d.qpos[3] = 1.0
        mujoco.mj_forward(self.mj_model, d)
        self._template_data = mjx.put_data(self.mj_model, d)

        self.collision_hull_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "collision_hull"
        )
        self.base_link_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        self.static_hfield_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_HFIELD, _STATIC_HFIELD_NAME
        )
        self.ground_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ground"
        )
        if self.config.static_obstacle_representation == "hfield":
            self.static_geom_names = ["ground"]
            self.static_geom_ids = jnp.array([self.ground_geom_id], dtype=jnp.int32)
        else:
            self.static_geom_names = static_obstacle_geom_names(self.static_field.positions.shape[0])
            self.static_geom_ids = jnp.array([
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in self.static_geom_names
            ], dtype=jnp.int32)

        if self.config.physical_dynamic_obstacles:
            self.dynamic_body_ids = jnp.array([
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in self.dynamic_body_names
            ], dtype=jnp.int32)
            self.dynamic_geom_ids = jnp.array([
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in self.dynamic_geom_names
            ], dtype=jnp.int32)
            self.dynamic_mocap_ids = jnp.array(
                np.asarray(self.mj_model.body_mocapid)[np.asarray(self.dynamic_body_ids)],
                dtype=jnp.int32,
            )
        else:
            self.dynamic_body_ids = jnp.zeros((0,), dtype=jnp.int32)
            self.dynamic_geom_ids = jnp.zeros((0,), dtype=jnp.int32)
            self.dynamic_mocap_ids = jnp.zeros((0,), dtype=jnp.int32)
        self.scene_version += 1

    def regenerate_static_obstacles(self, key: jnp.ndarray) -> int:
        """
        重新生成静态障碍物场（对应原代码每次运行重新生成地形）。

        用于训练多样性：每隔 regenerate_static_obs_interval 次迭代调用一次。
        注意：该方法会重建 `mj_model/mjx_model/_template_data/static_geom_ids`，调用方必须在
        JIT 边界外显式丢弃旧 `EnvState/FullEnvState`，并重新创建依赖旧场景的已编译闭包。

        Args:
            key: JAX PRNG key，用于生成新的随机障碍物布局

        Returns:
            当前场景版本号。每次重建后单调递增，可用于调试/测试 JIT 生命周期。
        """
        self.static_field = generate_static_obstacles(
            key,
            n_obstacles=self.config.n_static_obs,
            area_size=self.config.static_obs_map_range,
            generator_mode=self.config.static_terrain_generator,
            horizontal_scale=self.config.static_terrain_hscale,
            vertical_scale=self.config.static_terrain_vscale,
            platform_width=self.config.static_terrain_platform_width,
        )
        self._rebuild_physics_scene()
        return self.scene_version

    def reset(self, key: jnp.ndarray) -> Tuple[FullEnvState, ObsBundle]:
        """重置单个完整环境。"""
        (key, k_ss, k_sp, k_sh,
         k_ts, k_tp, k_th, k_dyn) = jax.random.split(key, 8)

        # 起点/目标（与简化环境相同）
        start_pos  = _sample_edge_position(k_ss, k_sp, k_sh,
                                            self.config.spawn_distance,
                                            self.config.spawn_edge_inset,
                                            self.config.height_range_min,
                                            self.config.height_range_max)
        target_pos = _sample_edge_position(k_ts, k_tp, k_th,
                                            self.config.spawn_distance,
                                            self.config.spawn_edge_inset,
                                            self.config.height_range_min,
                                            self.config.height_range_max)

        target_dir = target_pos - start_pos
        yaw = jnp.arctan2(target_dir[1], target_dir[0])
        rpy  = jnp.array([0.0, 0.0, yaw])
        quat = euler_to_quaternion(rpy)
        qpos = jnp.concatenate([start_pos, quat])
        qvel = jnp.zeros(6)

        dyn_obs = _reset_dynamic_obstacles(k_dyn, self.config)

        mjx_data = self._template_data.replace(qpos=qpos, qvel=qvel)
        if self.config.physical_dynamic_obstacles:
            mjx_data = _apply_dynamic_obstacles_to_data(mjx_data, dyn_obs, self.dynamic_mocap_ids)
        if self.config.lidar_use_scene_ray and (
            self.config.physical_static_obstacles or
            (self.config.physical_dynamic_obstacles and self.config.lidar_scan_dynamic)
        ):
            mjx_data = _prepare_lidar_scene_data(self.mjx_model, mjx_data)
        mjx_data = mjx.forward(self.mjx_model, mjx_data)

        min_h = jnp.minimum(start_pos[2], target_pos[2])
        max_h = jnp.maximum(start_pos[2], target_pos[2])

        state = FullEnvState(
            mjx_data=mjx_data,
            target_pos=target_pos,
            target_dir=target_dir,
            height_range=jnp.array([min_h, max_h]),
            prev_vel=jnp.zeros(3),
            step_count=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False),
            key=key,
            dyn_obs=dyn_obs,
        )

        root_state = _extract_root_state(mjx_data)
        obs = _get_obs_full(
            root_state, target_pos, target_dir, dyn_obs,
            self.static_field.positions, self.static_field.half_extents,
            self.config,
            self.mjx_model,
            mjx_data,
            self.base_link_body_id,
            _STATIC_GEOMGROUP_MASK,
            _DYNAMIC_GEOMGROUP_MASK,
            self.config.physical_dynamic_obstacles,
        )
        return state, obs

    def _fast_reset(self, key: jnp.ndarray) -> Tuple[FullEnvState, ObsBundle]:
        """快速重置单个完整环境（跳过 mjx.forward）。"""
        (key, k_ss, k_sp, k_sh,
         k_ts, k_tp, k_th, k_dyn) = jax.random.split(key, 8)

        start_pos  = _sample_edge_position(k_ss, k_sp, k_sh,
                                            self.config.spawn_distance,
                                            self.config.spawn_edge_inset,
                                            self.config.height_range_min,
                                            self.config.height_range_max)
        target_pos = _sample_edge_position(k_ts, k_tp, k_th,
                                            self.config.spawn_distance,
                                            self.config.spawn_edge_inset,
                                            self.config.height_range_min,
                                            self.config.height_range_max)

        target_dir = target_pos - start_pos
        yaw = jnp.arctan2(target_dir[1], target_dir[0])
        rpy  = jnp.array([0.0, 0.0, yaw])
        quat = euler_to_quaternion(rpy)
        qpos = jnp.concatenate([start_pos, quat])
        qvel = jnp.zeros(6)

        # 跳过 mjx.forward；若 LiDAR 依赖真实 scene ray，则至少刷新运动学量，
        # 以保证 autoreset 后首帧射线查询与当前 qpos 对齐。
        dyn_obs = _reset_dynamic_obstacles(k_dyn, self.config)

        mjx_data = self._template_data.replace(qpos=qpos, qvel=qvel)
        if self.config.physical_dynamic_obstacles:
            mjx_data = _apply_dynamic_obstacles_to_data(mjx_data, dyn_obs, self.dynamic_mocap_ids)
        if self.config.lidar_use_scene_ray and (
            self.config.physical_static_obstacles or
            (self.config.physical_dynamic_obstacles and self.config.lidar_scan_dynamic)
        ):
            mjx_data = _prepare_lidar_scene_data(self.mjx_model, mjx_data)

        min_h = jnp.minimum(start_pos[2], target_pos[2])
        max_h = jnp.maximum(start_pos[2], target_pos[2])

        state = FullEnvState(
            mjx_data=mjx_data,
            target_pos=target_pos,
            target_dir=target_dir,
            height_range=jnp.array([min_h, max_h]),
            prev_vel=jnp.zeros(3),
            step_count=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False),
            key=key,
            dyn_obs=dyn_obs,
        )

        # 分析计算初始观测（reset 时状态完全已知）
        root_state = jnp.concatenate([
            start_pos, quat, jnp.zeros(3), jnp.zeros(3),
        ])
        obs = _get_obs_full(
            root_state, target_pos, target_dir, dyn_obs,
            self.static_field.positions, self.static_field.half_extents,
            self.config,
            self.mjx_model,
            mjx_data,
            self.base_link_body_id,
            _STATIC_GEOMGROUP_MASK,
            _DYNAMIC_GEOMGROUP_MASK,
            self.config.physical_dynamic_obstacles,
        )
        return state, obs

    def step(
        self,
        state: FullEnvState,
        action: jnp.ndarray,
    ) -> Tuple[FullEnvState, ObsBundle, jnp.ndarray, jnp.ndarray, dict]:
        """完整环境前进一步。"""
        root_state = _extract_root_state(state.mjx_data)
        curr_vel   = root_state[7:10]

        # 动作变换 + Lee 控制器 + 物理推进
        target_dir_2d = state.target_dir.at[2].set(0.0)
        vel_cmd_world = vec_to_world(action, target_dir_2d)
        key, dyn_key = jax.random.split(state.key)
        new_step_count = state.step_count + 1
        new_dyn_obs = _step_dynamic_obstacles(
            state.dyn_obs, state.step_count, dyn_key, self.config,
        )

        ctrl = vel_to_mjx_ctrl(self.ctrl_params, root_state, vel_cmd_world)
        mjx_data = state.mjx_data.replace(ctrl=ctrl)
        if self.config.physical_dynamic_obstacles:
            mjx_data = _apply_dynamic_obstacles_to_data(mjx_data, new_dyn_obs, self.dynamic_mocap_ids)
        mjx_data = mjx.step(self.mjx_model, mjx_data)

        # 提取新状态
        new_root_state = _extract_root_state(mjx_data)

        # 完整观测
        obs = _get_obs_full(
            new_root_state, state.target_pos, state.target_dir,
            new_dyn_obs,
            self.static_field.positions, self.static_field.half_extents,
            self.config,
            self.mjx_model,
            mjx_data,
            self.base_link_body_id,
            _STATIC_GEOMGROUP_MASK,
            _DYNAMIC_GEOMGROUP_MASK,
            self.config.physical_dynamic_obstacles,
        )

        # 完整奖励（含安全奖励）
        reward = _compute_reward_full(
            new_root_state, state.target_pos,
            state.prev_vel, state.height_range,
            obs.lidar, new_dyn_obs,
            self.config,
        )

        static_contact = (
            _has_static_contact(mjx_data, self.collision_hull_geom_id, self.static_geom_ids)
            if self.config.physical_static_obstacles else None
        )
        dynamic_contact = (
            _has_dynamic_contact(mjx_data, self.collision_hull_geom_id, self.dynamic_geom_ids)
            if self.config.physical_dynamic_obstacles else None
        )

        # 完整终止条件
        terminated, truncated = _compute_done_full(
            new_root_state, new_step_count, obs.lidar, new_dyn_obs, self.config,
            static_contact=static_contact,
            dynamic_contact=dynamic_contact,
        )
        done = terminated | truncated

        # reach_goal stat（对应原代码 env.py:583）
        pos_new = new_root_state[:3]
        rpos_to_target = state.target_pos - pos_new
        reach_goal = jnp.linalg.norm(rpos_to_target) < self.config.reach_goal_dist

        # collision stat（对应原代码 env.py:570-571）
        if self.config.physical_static_obstacles:
            static_collision = _has_static_contact(
                mjx_data,
                self.collision_hull_geom_id,
                self.static_geom_ids,
            )
        else:
            static_collision = jnp.max(obs.lidar) > (
                self.config.lidar_range - self.config.collision_radius
            )
        dynamic_collision = (
            dynamic_contact
            if self.config.physical_dynamic_obstacles else
            _check_dynamic_collision(pos_new, new_dyn_obs, self.config)
        )
        collision = static_collision | dynamic_collision

        new_state = FullEnvState(
            mjx_data=mjx_data,
            target_pos=state.target_pos,
            target_dir=state.target_dir,
            height_range=state.height_range,
            prev_vel=new_root_state[7:10],  # 使用本步物理后速度（对应 B: self.prev_drone_vel_w = self.drone.vel_w）
            step_count=new_step_count,
            done=done,
            key=key,
            dyn_obs=new_dyn_obs,
        )

        info = {
            "step_count": new_step_count,
            "terminated": terminated,
            "truncated": truncated,
            "reach_goal": reach_goal,
            "collision": collision,
        }

        return new_state, obs, reward, done, info


# ============================================================
# 向量化完整环境封装
# ============================================================

class VectorizedFullNavigationEnv:
    """
    向量化完整导航环境（含障碍物和 LiDAR）。

    使用 jax.vmap 并行化 reset/step，支持自动重置。
    """

    def __init__(
        self,
        num_envs: int,
        config: EnvConfig = EnvConfig(),
        static_obs_seed: int = 42,
    ):
        self.num_envs = num_envs
        self.env = FullNavigationEnv(config, static_obs_seed)
        self.config = config

    def reset(self, key: jnp.ndarray) -> Tuple[FullEnvState, ObsBundle]:
        """重置所有环境。"""
        keys = jax.random.split(key, self.num_envs)
        return jax.vmap(self.env.reset)(keys)

    @property
    def scene_version(self) -> int:
        """返回底层物理静态场景版本号。"""
        return self.env.scene_version

    def regenerate_static_obstacles(self, key: jnp.ndarray) -> int:
        """
        重新生成静态障碍物场，代理到内部 FullNavigationEnv。
        该操作会替换底层 `mjx.Model`；调用方需要在 Python 层重置环境状态并重建 JIT 闭包。
        """
        return self.env.regenerate_static_obstacles(key)

    def step(
        self,
        states: FullEnvState,
        actions: jnp.ndarray,
    ) -> Tuple[FullEnvState, ObsBundle, jnp.ndarray, jnp.ndarray, dict]:
        """所有环境前进一步。"""
        return jax.vmap(self.env.step)(states, actions)

    def step_with_autoreset(
        self,
        states: FullEnvState,
        actions: jnp.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[FullEnvState, ObsBundle, jnp.ndarray, jnp.ndarray, dict]:
        """
        带自动重置的步进。

        done 的环境自动 reset，返回新 episode 的初始观测（ObsBundle）。
        注：dyn_step_count 跨 episode 不重置（对应 B 的全局计数器行为）。
        """
        new_states, obs, rewards, dones, infos = self.step(states, actions)

        reset_keys = jax.random.split(key, self.num_envs)
        reset_states, reset_obs = jax.vmap(self.env._fast_reset)(reset_keys)

        # 根据 done 条件选择状态
        def _select(reset_val, continue_val):
            if reset_val.ndim <= 1:
                return jnp.where(dones, reset_val, continue_val)
            else:
                shape = [dones.shape[0]] + [1] * (reset_val.ndim - 1)
                return jnp.where(dones.reshape(shape), reset_val, continue_val)

        # FullEnvState 字段（含嵌套 DynObsState）
        final_states = jax.tree.map(_select, reset_states, new_states)

        # 保留 dyn_step_count 不受 autoreset 影响（全局计数器，对应 B: dyn_obs_step_count）
        # reset_states 的 dyn_step_count=0，但我们需要保持 new_states 中的累计值
        final_dyn_obs = final_states.dyn_obs._replace(
            dyn_step_count=new_states.dyn_obs.dyn_step_count,
        )
        final_states = final_states._replace(dyn_obs=final_dyn_obs)

        # ObsBundle 字段
        final_obs = jax.tree.map(
            lambda r, c: _select(r, c),
            reset_obs, obs,
        )

        return final_states, final_obs, rewards, dones, infos
