"""
training/eval.py
MJX + JAX 策略评估脚本

迁移自: training/scripts/eval.py

使用方式：
  cd /home/xuk/workspace/drone_navrl/mjx_navrl
  python training/eval.py --checkpoint checkpoints/checkpoint_best.pkl
  python training/eval.py --checkpoint checkpoints/checkpoint_best.pkl --render
  python training/eval.py --checkpoint checkpoints/checkpoint_best.pkl --num_episodes 100

  # 自定义导航任务（指定起点和目标）
  python training/eval.py --checkpoint checkpoints/checkpoint_best.pkl --render \
    --start 10 0 1.5 --target -10 0 1.5

  # 预定义导航任务
  python training/eval.py --checkpoint checkpoints/checkpoint_best.pkl --render --task short
  python training/eval.py --checkpoint checkpoints/checkpoint_best.pkl --render --task long
"""

import os
import sys
import time
import argparse
import pickle
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from envs.navigation_env import (
    NavigationEnv,
    FullNavigationEnv,
    VectorizedNavigationEnv,
    VectorizedFullNavigationEnv,
    EnvConfig,
    EnvState,
    FullEnvState,
    _extract_root_state,
    _get_obs,
    _get_obs_full,
    _apply_dynamic_obstacles_to_data,
    _STATIC_GEOMGROUP_MASK,
    _DYNAMIC_GEOMGROUP_MASK,
    _prepare_lidar_scene_data,
    _reset_dynamic_obstacles,
)
from networks.actor_critic import ActorCritic, ActorCriticFull, beta_mode
from training.ppo import PPOConfig
from training.train import (
    load_config as load_train_config,
    load_checkpoint as load_train_checkpoint,
    make_env_config,
    make_ppo_config,
    checkpoint_uses_full_obs,
)
from utils.math_utils import euler_to_quaternion


# ============================================================
# 预定义导航任务
# ============================================================

TASK_PRESETS = {
    'short':    {'start': [5, 0, 1.5],     'target': [-5, 0, 1.5]},
    'medium':   {'start': [12, 0, 1.5],    'target': [-12, 0, 1.5]},
    'long':     {'start': [20, 0, 1.5],    'target': [-20, 0, 1.5]},
    'vertical': {'start': [0, 0, 0.5],     'target': [0, 0, 3.0]},
    'diagonal': {'start': [10, 10, 0.5],   'target': [-10, -10, 2.5]},
    'return':   {'start': [-8, 0, 1.5],    'target': [8, 0, 1.5]},
    'cross':    {'start': [18, -18, 1.5],  'target': [-18, 18, 1.5]},
    'descent':  {'start': [0, 0, 3.2],     'target': [0, 0, 0.8]},
    'far':      {'start': [22, 8, 1.0],    'target': [-22, -8, 2.2]},
}


# ============================================================
# 加载检查点
# ============================================================

def load_checkpoint(path: str) -> dict:
    """加载训练检查点。"""
    return load_train_checkpoint(path)


def _checkpoint_uses_full_obs(checkpoint: dict) -> bool:
    """兼容旧接口：优先读取 manifest，再回退到参数树启发式推断。"""
    return checkpoint_uses_full_obs(checkpoint)


def _validate_custom_task(
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    env_config: EnvConfig,
) -> tuple:
    """校验并规范化自定义任务输入。"""
    start = np.asarray(start_pos, dtype=np.float32).reshape(-1)
    target = np.asarray(target_pos, dtype=np.float32).reshape(-1)
    if start.shape != (3,) or target.shape != (3,):
        raise ValueError("起点和目标都必须是 3D 坐标")

    warnings = []
    z_min = env_config.height_min_terminate + 0.05
    z_max = env_config.height_max_terminate - 0.05

    clipped_start_z = float(np.clip(start[2], z_min, z_max))
    clipped_target_z = float(np.clip(target[2], z_min, z_max))
    if clipped_start_z != float(start[2]):
        warnings.append(
            f"起点高度 {start[2]:.2f}m 超出安全范围，已裁剪到 {clipped_start_z:.2f}m"
        )
        start[2] = clipped_start_z
    if clipped_target_z != float(target[2]):
        warnings.append(
            f"目标高度 {target[2]:.2f}m 超出安全范围，已裁剪到 {clipped_target_z:.2f}m"
        )
        target[2] = clipped_target_z

    travel_dist = float(np.linalg.norm(target - start))
    if travel_dist < 1e-3:
        raise ValueError("起点与目标不能重合")
    if travel_dist < 1.0:
        warnings.append("起点与目标距离过近，任务演示可能在几步内结束")

    xy_limit = env_config.spawn_distance * 1.25
    if np.any(np.abs(start[:2]) > xy_limit) or np.any(np.abs(target[:2]) > xy_limit):
        warnings.append(
            f"任务位置超出常规训练区域 ±{env_config.spawn_distance:.1f}m，策略泛化表现可能下降"
        )

    return start.astype(np.float32), target.astype(np.float32), warnings


# ============================================================
# 批量评估
# ============================================================

def evaluate_batch(
    params: dict,
    network,
    env,
    config: PPOConfig,
    key: jnp.ndarray,
    num_steps: int = 2200,
) -> dict:
    """
    批量确定性评估。

    支持简化（VectorizedNavigationEnv + ActorCritic）和
    完整（VectorizedFullNavigationEnv + ActorCriticFull）两种模式。
    通过 config.use_full_obs 自动切换。

    使用 jax.lax.scan 在设备上一次性完成所有步，避免逐步 GPU→CPU 同步。

    Args:
        params:    Flax 模型参数
        network:   ActorCritic 或 ActorCriticFull 网络实例
        env:       VectorizedNavigationEnv 或 VectorizedFullNavigationEnv
        config:    PPO 配置
        key:       PRNG key
        num_steps: 评估步数

    Returns:
        dict: 评估统计
    """
    key, reset_key = jax.random.split(key)
    env_states, obs = jax.jit(env.reset)(reset_key)

    @jax.jit
    def _eval_rollout(env_states, obs, key):
        def _scan_step(carry, _):
            env_states, obs, key = carry
            # 网络前向（确定性：使用 Beta 众数）
            if config.use_full_obs:
                alpha, beta_param, value = jax.vmap(
                    lambda s, l, d, dy: network.apply(params, s, l, d, dy)
                )(obs.state, obs.lidar, obs.direction, obs.dynamic_obs)
            else:
                alpha, beta_param, value = network.apply(params, obs)
            action_normalized = beta_mode(alpha, beta_param)
            action_goal = 2.0 * action_normalized * config.action_limit - config.action_limit

            key, env_key = jax.random.split(key)
            new_env_states, new_obs, rewards, dones, infos = env.step_with_autoreset(
                env_states, action_goal, env_key,
            )
            return (new_env_states, new_obs, key), (rewards, dones)

        _, (all_rewards, all_dones) = jax.lax.scan(
            _scan_step,
            (env_states, obs, key),
            None,
            length=num_steps,
        )
        return all_rewards, all_dones  # (num_steps, N)

    print(f"  评估 {config.num_envs} 个环境，{num_steps} 步...")

    all_rewards, all_dones = _eval_rollout(env_states, obs, key)

    # 单次 GPU→CPU 传输
    rewards_np = np.array(all_rewards)
    dones_np = np.array(all_dones).astype(bool)

    # CPU 端 episode 统计追踪
    episode_rewards = []
    episode_lengths = []
    current_rewards = np.zeros(config.num_envs, dtype=np.float32)
    current_lengths = np.zeros(config.num_envs, dtype=np.int32)

    for t in range(num_steps):
        current_rewards += rewards_np[t]
        current_lengths += 1
        done_mask = dones_np[t]
        n_done = done_mask.sum()
        if n_done > 0:
            episode_rewards.extend(current_rewards[done_mask].tolist())
            episode_lengths.extend(current_lengths[done_mask].tolist())
            current_rewards[done_mask] = 0.0
            current_lengths[done_mask] = 0

    results = {
        'num_episodes': len(episode_rewards),
        'mean_reward': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        'std_reward': float(np.std(episode_rewards)) if episode_rewards else 0.0,
        'mean_length': float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        'std_length': float(np.std(episode_lengths)) if episode_lengths else 0.0,
        'min_reward': float(np.min(episode_rewards)) if episode_rewards else 0.0,
        'max_reward': float(np.max(episode_rewards)) if episode_rewards else 0.0,
    }
    return results


# ============================================================
# 自定义导航任务重置
# ============================================================

def _custom_reset(
    env: NavigationEnv,
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    key: jnp.ndarray,
) -> tuple:
    """
    使用指定的起点和目标位置重置环境（跳过随机采样）。

    Args:
        env:        NavigationEnv 实例
        start_pos:  (3,) 起点世界坐标
        target_pos: (3,) 目标世界坐标
        key:        JAX PRNG key

    Returns:
        (EnvState, obs): 初始状态和 8D 观测
    """
    import mujoco.mjx as mjx

    start_pos_j = jnp.array(start_pos, dtype=jnp.float32)
    target_pos_j = jnp.array(target_pos, dtype=jnp.float32)
    target_dir = target_pos_j - start_pos_j
    yaw = jnp.arctan2(target_dir[1], target_dir[0])

    rpy = jnp.array([0.0, 0.0, yaw])
    quat = euler_to_quaternion(rpy)
    qpos = jnp.concatenate([start_pos_j, quat])
    qvel = jnp.zeros(6)

    mjx_data = env._template_data.replace(qpos=qpos, qvel=qvel)
    mjx_data = mjx.forward(env.mjx_model, mjx_data)

    min_h = jnp.minimum(start_pos_j[2], target_pos_j[2])
    max_h = jnp.maximum(start_pos_j[2], target_pos_j[2])
    height_range = jnp.array([min_h, max_h])

    state = EnvState(
        mjx_data=mjx_data,
        target_pos=target_pos_j,
        target_dir=target_dir,
        height_range=height_range,
        prev_vel=jnp.zeros(3),
        step_count=jnp.array(0, dtype=jnp.int32),
        done=jnp.array(False),
        key=key,
    )

    root_state = _extract_root_state(mjx_data)
    obs = _get_obs(root_state, target_pos_j, target_dir)
    return state, obs


def _custom_reset_full(
    env: FullNavigationEnv,
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    key: jnp.ndarray,
) -> tuple:
    """完整多模态环境的自定义任务重置。"""
    import mujoco.mjx as mjx

    start_pos_j = jnp.array(start_pos, dtype=jnp.float32)
    target_pos_j = jnp.array(target_pos, dtype=jnp.float32)
    target_dir = target_pos_j - start_pos_j
    yaw = jnp.arctan2(target_dir[1], target_dir[0])

    rpy = jnp.array([0.0, 0.0, yaw])
    quat = euler_to_quaternion(rpy)
    qpos = jnp.concatenate([start_pos_j, quat])
    qvel = jnp.zeros(6)

    key, dyn_key = jax.random.split(key)
    dyn_obs = _reset_dynamic_obstacles(dyn_key, env.config)

    mjx_data = env._template_data.replace(qpos=qpos, qvel=qvel)
    if env.config.physical_dynamic_obstacles:
        mjx_data = _apply_dynamic_obstacles_to_data(mjx_data, dyn_obs, env.dynamic_mocap_ids)
    if env.config.lidar_use_scene_ray and env.config.physical_static_obstacles:
        mjx_data = _prepare_lidar_scene_data(env.mjx_model, mjx_data)
    mjx_data = mjx.forward(env.mjx_model, mjx_data)

    min_h = jnp.minimum(start_pos_j[2], target_pos_j[2])
    max_h = jnp.maximum(start_pos_j[2], target_pos_j[2])

    state = FullEnvState(
        mjx_data=mjx_data,
        target_pos=target_pos_j,
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
        root_state,
        target_pos_j,
        target_dir,
        dyn_obs,
        env.static_field.positions,
        env.static_field.half_extents,
        env.config,
        env.mjx_model,
        mjx_data,
        env.base_link_body_id,
        _STATIC_GEOMGROUP_MASK,
        _DYNAMIC_GEOMGROUP_MASK,
        env.config.physical_dynamic_obstacles,
    )
    return state, obs


# ============================================================
# 目标位置可视化标记
# ============================================================

def _add_target_marker(viewer, target_pos: np.ndarray):
    """
    在 viewer 场景中添加目标位置的可视化标记（红色球体 + 绿色垂直线）。

    Args:
        viewer:     MuJoCo passive viewer
        target_pos: (3,) 目标世界坐标
    """
    # 红色半透明球体标记目标位置
    viewer.user_scn.ngeom = 0  # 清除旧标记
    mj_geom = viewer.user_scn.geoms[0]
    import mujoco
    mujoco.mjv_initGeom(
        mj_geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.5, 0.5, 0.5],
        pos=target_pos.astype(np.float64),
        mat=np.eye(3).flatten().astype(np.float64),
        rgba=np.array([0.9, 0.1, 0.1, 0.6], dtype=np.float32),
    )
    viewer.user_scn.ngeom = 1

    # 绿色垂直线标记（从地面到目标高度 + 2m）
    if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
        mj_geom2 = viewer.user_scn.geoms[1]
        line_bottom = np.array([target_pos[0], target_pos[1], 0.0], dtype=np.float64)
        line_top = np.array([target_pos[0], target_pos[1], target_pos[2] + 2.0], dtype=np.float64)
        mujoco.mjv_initGeom(
            mj_geom2,
            type=mujoco.mjtGeom.mjGEOM_LINE,
            size=[0.02, 0, 0],
            pos=(line_bottom + line_top) / 2.0,
            mat=np.eye(3).flatten().astype(np.float64),
            rgba=np.array([0.1, 0.9, 0.1, 0.8], dtype=np.float32),
        )
        mujoco.mjv_connector(
            mj_geom2,
            mujoco.mjtGeom.mjGEOM_LINE,
            3.0,
            line_bottom,
            line_top,
        )
        mj_geom2.rgba[:] = np.array([0.1, 0.9, 0.1, 0.8], dtype=np.float32)
        viewer.user_scn.ngeom = 2


def _render_trajectory(viewer, trajectory: list, geom_offset: int = 2):
    """
    在 viewer 场景中渲染飞行轨迹（蓝色半透明球体轨迹线）。

    Args:
        viewer:      MuJoCo passive viewer
        trajectory:  list of (3,) np.ndarray 位置点
        geom_offset: user_scn.geoms 中的起始索引（0-1 被目标标记占用）
    """
    import mujoco

    max_trail = min(len(trajectory), viewer.user_scn.maxgeom - geom_offset)
    if max_trail <= 0:
        return

    # 只渲染最近的 max_trail 个点
    points = trajectory[-max_trail:]
    for i, pos in enumerate(points):
        idx = geom_offset + i
        mj_geom = viewer.user_scn.geoms[idx]
        # 颜色渐变：旧的点更透明
        alpha = 0.2 + 0.5 * (i / max(max_trail - 1, 1))
        mujoco.mjv_initGeom(
            mj_geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.06, 0.06, 0.06],
            pos=pos.astype(np.float64),
            mat=np.eye(3).flatten().astype(np.float64),
            rgba=np.array([0.2, 0.4, 0.9, alpha], dtype=np.float32),
        )
    viewer.user_scn.ngeom = geom_offset + max_trail


def _viewer_lock(viewer):
    """兼容 passive viewer 的锁接口。"""
    return viewer.lock() if hasattr(viewer, 'lock') else nullcontext()


def _viewer_is_running(viewer) -> bool:
    """兼容不同 MuJoCo 版本的 viewer 运行状态查询。"""
    return viewer.is_running() if hasattr(viewer, 'is_running') else True


def _sync_viewer_scene(
    viewer,
    viewer_data,
    model,
    mjx_data,
    target_pos: np.ndarray,
    trajectory: list,
):
    """在加锁条件下同步物理状态与用户场景几何。"""
    import mujoco

    with _viewer_lock(viewer):
        viewer_data.qpos[:] = np.array(mjx_data.qpos)
        viewer_data.qvel[:] = np.array(mjx_data.qvel)
        mujoco.mj_forward(model, viewer_data)
        _add_target_marker(viewer, target_pos)
        _render_trajectory(viewer, trajectory)
    viewer.sync()


# ============================================================
# 单环境渲染评估（MuJoCo Viewer + 相机跟踪）
# ============================================================

def evaluate_render(
    params: dict,
    network,
    config: PPOConfig,
    env_config: EnvConfig,
    key: jnp.ndarray,
    num_episodes: int = 5,
    custom_start: np.ndarray = None,
    custom_target: np.ndarray = None,
    terminate_on_goal: bool = True,
):
    """
    使用 MuJoCo Viewer 渲染评估，相机自动跟踪无人机。

    需要图形化环境（非 headless）。

    Args:
        params:        Flax 模型参数
        network:       ActorCritic 网络实例
        config:        PPO 配置
        key:           PRNG key
        num_episodes:  评估 episode 数
        custom_start:  (3,) 自定义起点位置，None 则随机
        custom_target: (3,) 自定义目标位置，None 则随机
    """
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("  Error: mujoco viewer not available")
        return

    env = FullNavigationEnv(env_config) if config.use_full_obs else NavigationEnv(env_config)

    if config.use_full_obs:
        @jax.jit
        def _get_action(params, obs, target_dir):
            target_dir_2d = target_dir.at[2].set(0.0)
            direction = target_dir_2d / (jnp.linalg.norm(target_dir_2d) + 1e-7)
            alpha, beta_param, value = network.apply(
                params, obs.state, obs.lidar, obs.direction, obs.dynamic_obs,
            )
            action = beta_mode(alpha, beta_param)
            action_goal = 2.0 * action * config.action_limit - config.action_limit
            return action_goal
    else:
        @jax.jit
        def _get_action(params, obs):
            alpha, beta_param, value = network.apply(params, obs)
            action = beta_mode(alpha, beta_param)
            action_goal = 2.0 * action * config.action_limit - config.action_limit
            return action_goal

    @jax.jit
    def _env_reset(key):
        return env.reset(key)

    @jax.jit
    def _env_step(state, action):
        return env.step(state, action)

    use_custom = custom_start is not None and custom_target is not None

    print(f"\n  渲染评估 {num_episodes} 个 episodes...")
    if use_custom:
        print(f"  自定义导航任务: {custom_start} -> {custom_target}")
        dist = np.linalg.norm(custom_target - custom_start)
        print(f"  起点-目标距离: {dist:.2f}m")
    if config.use_full_obs:
        print("  渲染环境: 完整多模态 + 真实静态障碍物物理场景")
    else:
        print("  渲染环境: 简化 8D")

    # JIT 预编译
    print("  JIT 编译中（首次运行需要等待）...")
    key, warmup_key = jax.random.split(key)
    _warmup_state, _warmup_obs = _env_reset(warmup_key)
    if config.use_full_obs:
        _warmup_action = _get_action(params, _warmup_obs, _warmup_state.target_dir)
    else:
        _warmup_action = _get_action(params, _warmup_obs)
    _ = _env_step(_warmup_state, _warmup_action)
    print("  JIT 编译完成")

    viewer_data = mujoco.MjData(env.mj_model)
    with mujoco.viewer.launch_passive(env.mj_model, viewer_data) as viewer:

        # ---- 初始化相机参数 ----
        # 跟踪模式：相机自动跟踪无人机 body，无需手动更新 lookat
        body_id = mujoco.mj_name2id(
            env.mj_model, mujoco.mjtObj.mjOBJ_BODY, 'base_link'
        )
        with _viewer_lock(viewer):
            cam = viewer.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.fixedcamid = -1
            cam.trackbodyid = body_id
            cam.distance = 8.0 if use_custom else 10.0
            cam.azimuth = 140.0
            cam.elevation = -24.0

        for ep in range(num_episodes):
            if not _viewer_is_running(viewer):
                print("  Viewer 已关闭，提前结束渲染评估")
                return

            key, ep_key = jax.random.split(key)

            # 重置环境（自定义或随机）
            if use_custom:
                if config.use_full_obs:
                    state, obs = _custom_reset_full(env, custom_start, custom_target, ep_key)
                else:
                    state, obs = _custom_reset(env, custom_start, custom_target, ep_key)
            else:
                state, obs = _env_reset(ep_key)

            ep_reward = 0.0
            ep_steps = 0
            info = {'terminated': False, 'truncated': False, 'reach_goal': False, 'collision': False}

            # 获取目标位置用于显示
            target_pos_np = np.array(state.target_pos)
            drone_pos_np = np.array(state.mjx_data.qpos[:3])
            print(f"\n  Episode {ep + 1}: 起点={drone_pos_np.round(2)}, "
                  f"目标={target_pos_np.round(2)}, "
                  f"距离={np.linalg.norm(target_pos_np - drone_pos_np):.2f}m")

            reach_goal = False
            trajectory = []  # 飞行轨迹点
            _sync_viewer_scene(
                viewer, viewer_data, env.mj_model, state.mjx_data, target_pos_np, trajectory,
            )

            while not bool(state.done) and ep_steps < env.config.max_episode_length:
                if not _viewer_is_running(viewer):
                    print("  Viewer 已关闭，提前结束渲染评估")
                    return

                if config.use_full_obs:
                    action = _get_action(params, obs, state.target_dir)
                else:
                    action = _get_action(params, obs)
                state, obs, reward, done, info = _env_step(state, action)
                ep_reward += float(reward)
                ep_steps += 1

                # 收集轨迹点（每 10 步记录一次，限制最大 300 点）
                drone_pos = np.array(state.mjx_data.qpos[:3])
                if ep_steps % 10 == 0:
                    trajectory.append(drone_pos.copy())
                    if len(trajectory) > 300:
                        trajectory.pop(0)

                # 渲染目标标记 + 飞行轨迹
                _sync_viewer_scene(
                    viewer, viewer_data, env.mj_model, state.mjx_data, target_pos_np, trajectory,
                )
                time.sleep(0.016)

                # 检查是否到达目标
                if bool(info['reach_goal']):
                    reach_goal = True
                    if use_custom and terminate_on_goal:
                        print(f"    [Step {ep_steps:4d}] 已到达目标，提前结束当前任务")
                        break

                # 实时状态显示（每 50 步输出一次）
                if ep_steps % 50 == 0:
                    dist_to_target = np.linalg.norm(target_pos_np - drone_pos)
                    vel = np.linalg.norm(np.array(state.mjx_data.qvel[:3]))
                    print(f"    [Step {ep_steps:4d}] dist={dist_to_target:.2f}m | "
                          f"vel={vel:.2f}m/s | pos=({drone_pos[0]:.1f}, {drone_pos[1]:.1f}, {drone_pos[2]:.1f})")

            status = (
                "REACHED" if reach_goal else
                ("COLLISION" if bool(info.get('collision', False)) else
                 ("TERMINATED" if bool(info.get('terminated', False)) else "TRUNCATED"))
            )
            final_dist = np.linalg.norm(np.array(state.target_pos) - np.array(state.mjx_data.qpos[:3]))
            flight_time = ep_steps * env.config.sim_dt
            print(f"  Episode {ep + 1}: reward={ep_reward:.2f}, steps={ep_steps}, "
                  f"time={flight_time:.2f}s, status={status}, final_dist={final_dist:.2f}m")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MJX NavRL Policy Evaluation")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='检查点文件路径')
    parser.add_argument('--config', type=str, default=None,
                        help='训练配置文件路径，默认使用 configs/default.yaml')
    parser.add_argument('--route', type=str, default=None, choices=['8d', 'full'],
                        help='正式评估路线预设：8d 或 full（会在加载配置后覆盖关键开关）')
    parser.add_argument('--num_envs', type=int, default=64,
                        help='评估并行环境数')
    parser.add_argument('--num_steps', type=int, default=2200,
                        help='评估步数')
    parser.add_argument('--render', action='store_true',
                        help='使用 MuJoCo Viewer 渲染')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='渲染评估 episode 数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--use_full_obs', action='store_true',
                        help='加载完整多模态观测检查点（含 LiDAR + 障碍物）')
    parser.add_argument('--start', type=float, nargs=3, default=None,
                        metavar=('X', 'Y', 'Z'),
                        help='自定义起点位置 (x y z)，需配合 --render 使用')
    parser.add_argument('--target', type=float, nargs=3, default=None,
                        metavar=('X', 'Y', 'Z'),
                        help='自定义目标位置 (x y z)，需配合 --render 使用')
    parser.add_argument('--task', type=str, default=None,
                        choices=list(TASK_PRESETS.keys()),
                        help='预定义导航任务，需配合 --render 使用')
    args = parser.parse_args()

    print("=" * 60)
    print("MJX NavRL Policy Evaluation")
    print("=" * 60)

    # 加载检查点
    print(f"\n  加载检查点: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint)
    params = jax.device_put(checkpoint['params'])
    manifest = checkpoint.get('manifest', {})

    # 根据训练配置和检查点结构恢复评估配置
    manifest_route = manifest.get('route') if isinstance(manifest, dict) else None
    cfg = load_train_config(args.config, route=args.route or manifest_route)
    config = make_ppo_config(cfg, args)
    env_config = make_env_config(cfg)
    checkpoint_use_full_obs = _checkpoint_uses_full_obs(checkpoint)

    if manifest:
        manifest_ppo = manifest.get('ppo_config', {})
        manifest_env = manifest.get('env_config', {})
        if manifest_env:
            env_config = EnvConfig(**manifest_env)
        if manifest_ppo:
            restored_ppo = dict(manifest_ppo)
            restored_ppo['num_envs'] = args.num_envs
            config = PPOConfig(**restored_ppo)

    route_name = args.route.lower() if args.route else manifest_route
    if route_name:
        print(f"  评估路线: {route_name}")

    if checkpoint_use_full_obs and not config.use_full_obs:
        print("  自动检测到完整多模态检查点，启用 `use_full_obs=True`")
        config = config._replace(use_full_obs=True)
    elif config.use_full_obs and not checkpoint_use_full_obs:
        print("  Warning: 检查点不包含多模态分支参数，回退到简化 8D 策略")
        config = config._replace(use_full_obs=False)

    if args.use_full_obs:
        print("  CLI 请求: 完整多模态观测")

    if config.use_full_obs:
        network = ActorCriticFull(action_dim=config.action_dim)
        print("  观测模式: 完整多模态 (LiDAR + 障碍物)")
    else:
        network = ActorCritic(action_dim=config.action_dim)
        print("  观测模式: 简化 8D")

    key = jax.random.PRNGKey(args.seed)

    if args.render:
        # 自定义导航任务参数
        custom_start = np.array(args.start, dtype=np.float32) if args.start else None
        custom_target = np.array(args.target, dtype=np.float32) if args.target else None

        # --task 预设（--start/--target 显式参数优先）
        if args.task and custom_start is None and custom_target is None:
            preset = TASK_PRESETS[args.task]
            custom_start = np.array(preset['start'], dtype=np.float32)
            custom_target = np.array(preset['target'], dtype=np.float32)
            dist = np.linalg.norm(custom_target - custom_start)
            print(f"  导航任务预设: {args.task} ({dist:.1f}m)")

        if (custom_start is None) != (custom_target is None):
            print("  Warning: --start 和 --target 需同时指定，忽略自定义任务")
            custom_start = custom_target = None
        elif custom_start is not None and custom_target is not None:
            try:
                custom_start, custom_target, task_warnings = _validate_custom_task(
                    custom_start, custom_target, env_config,
                )
                for warning in task_warnings:
                    print(f"  Warning: {warning}")
            except ValueError as exc:
                print(f"  Error: {exc}")
                return

        evaluate_render(
            params,
            network,
            config,
            env_config,
            key,
            args.num_episodes,
            custom_start=custom_start,
            custom_target=custom_target,
            terminate_on_goal=True,
        )
    else:
        # 批量评估
        if config.use_full_obs:
            env = VectorizedFullNavigationEnv(args.num_envs, env_config)
        else:
            env = VectorizedNavigationEnv(args.num_envs, env_config)

        results = evaluate_batch(
            params, network, env, config, key, args.num_steps,
        )

        print("\n" + "=" * 60)
        print("评估结果")
        print("=" * 60)
        print(f"  完成 episodes: {results['num_episodes']}")
        print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  奖励范围: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
        print(f"  平均步数: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
