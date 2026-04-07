"""
controllers/lee_controller.py
Lee 位置控制器的 JAX 实现（纯函数式，支持 jit/vmap）

迁移自: third_party/OmniDrones/omni_drones/controllers/lee_position_controller.py
参数来源:
  - 无人机参数: third_party/OmniDrones/omni_drones/robots/assets/usd/hummingbird.yaml
  - 控制器增益: third_party/OmniDrones/omni_drones/controllers/cfg/lee_controller_hummingbird.yaml

原理：
  Lee 控制器 (https://arxiv.org/abs/1003.2005) 通过以下步骤将速度指令转为电机推力：
  1. 从速度误差计算期望加速度
  2. 从期望加速度方向推导期望姿态 R_des
  3. 计算 SO(3) 上的姿态误差
  4. PD 控制器生成角加速度指令
  5. 混合矩阵(mixer)将 [角加速度, 总推力] 映射到 4 个电机指令

速度跟踪模式：
  在原 env.py 中，PPO 输出 3D 速度指令，LeePositionController 以
  target_pos = 当前位置（即位置误差=0）的方式进行速度跟踪。
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.math_utils import (
    quat_to_rotation_matrix,
    quat_rotate_inverse,
    quaternion_to_euler,
)


# ============================================================
# 控制器参数（不可变，JAX pytree 兼容）
# ============================================================

class LeeControllerParams(NamedTuple):
    """
    Lee 位置控制器的所有预计算常数。
    使用 NamedTuple 使其成为 JAX pytree leaf，可在 jit/vmap 中使用。
    """
    pos_gain: jnp.ndarray        # (3,) 位置增益
    vel_gain: jnp.ndarray        # (3,) 速度增益
    attitude_gain: jnp.ndarray   # (3,) 姿态增益（已乘以惯量矩阵逆）
    ang_rate_gain: jnp.ndarray   # (3,) 角速率增益（已乘以惯量矩阵逆）
    mixer: jnp.ndarray           # (4, 4) 混合矩阵
    max_thrusts: jnp.ndarray     # (4,) 每电机最大推力
    mass: jnp.ndarray            # () 无人机质量
    gravity: jnp.ndarray         # (3,) 重力向量 [0, 0, g]


class LeeDebugInfo(NamedTuple):
    """控制器关键中间量（用于低频观测与问题定位）。"""
    vel_error: jnp.ndarray       # (3,)
    acc: jnp.ndarray             # (3,)
    target_yaw: jnp.ndarray      # (1,)
    ang_vel_body: jnp.ndarray    # (3,)
    ang_error: jnp.ndarray       # (3,)
    ang_acc: jnp.ndarray         # (3,)
    thrust: jnp.ndarray          # (1,)
    cmd_norm: jnp.ndarray        # (4,) [-1, 1]
    ctrl_raw: jnp.ndarray        # (4,) [0, max]
    ctrl_clipped: jnp.ndarray    # (4,) [0, max]
    clip_mask: jnp.ndarray       # (4,) bool


# ============================================================
# 混合矩阵计算（从旋翼配置推导）
# ============================================================

def _compute_mixer(
    rotor_angles: jnp.ndarray,
    arm_lengths: jnp.ndarray,
    force_constants: jnp.ndarray,
    moment_constants: jnp.ndarray,
    directions: jnp.ndarray,
    inertia_diag: jnp.ndarray,
) -> jnp.ndarray:
    """
    计算混合矩阵 mixer，将 [τ_x, τ_y, τ_z, F_total] 映射到 4 个电机指令。

    迁移自 lee_position_controller.py: compute_parameters()

        A = [[sin(θ_i) * l_i],     — 滚转力矩臂
            [-cos(θ_i) * l_i],    — 俯仰力矩臂
            [d_i * k_m / k_f],    — 偏航反应扭矩系数（直接使用执行器 gear[5] 符号）
         [1, 1, 1, 1]]         — 总推力

    mixer = A^T (A A^T)^{-1} I_diag

    其中 I_diag = diag(Ixx, Iyy, Izz, 1)
    """
    A = jnp.stack([
        jnp.sin(rotor_angles) * arm_lengths,
        -jnp.cos(rotor_angles) * arm_lengths,
        directions * moment_constants / force_constants,
        jnp.ones_like(rotor_angles),
    ])  # (4, 4)

    I_diag = jnp.diag(jnp.concatenate([inertia_diag, jnp.array([1.0])]))  # (4, 4)
    mixer = A.T @ jnp.linalg.inv(A @ A.T) @ I_diag  # (4, 4)
    return mixer


# ============================================================
# 工厂函数：创建 Hummingbird 控制器参数
# ============================================================

def create_lee_params() -> LeeControllerParams:
    """
    创建 Hummingbird 无人机的 Lee 控制器参数。

    所有数值硬编码自：
      - hummingbird.yaml: 物理参数
      - lee_controller_hummingbird.yaml: 控制器增益
    """
    # ---- 旋翼配置 (hummingbird.yaml) ----
    rotor_angles = jnp.array([0.0, 1.5707963268, 3.1415926536, -1.5707963268])
    arm_lengths = jnp.array([0.17, 0.17, 0.17, 0.17])
    force_constants = jnp.array([8.54858e-06] * 4)
    moment_constants = jnp.array([1.3677728816219314e-07] * 4)
    directions = jnp.array([-1.0, 1.0, -1.0, 1.0])
    max_rot_vel = jnp.array([838.0, 838.0, 838.0, 838.0])

    # ---- 惯量 (hummingbird.yaml) ----
    inertia_diag = jnp.array([0.007, 0.007, 0.012])  # Ixx, Iyy, Izz
    I_3x3_inv = jnp.diag(1.0 / inertia_diag)         # (3, 3)

    # ---- 控制器增益 (lee_controller_hummingbird.yaml) ----
    position_gain = jnp.array([4.0, 4.0, 4.0])
    velocity_gain = jnp.array([2.2, 2.2, 2.2])
    attitude_gain_raw = jnp.array([0.7, 0.7, 0.035])
    angular_rate_gain_raw = jnp.array([0.1, 0.1, 0.025])

    # 增益预乘以惯量矩阵逆（与原代码 lee_position_controller.py:108-112 一致）
    attitude_gain = attitude_gain_raw @ I_3x3_inv      # [100.0, 100.0, 2.917]
    ang_rate_gain = angular_rate_gain_raw @ I_3x3_inv   # [14.286, 14.286, 2.083]

    # ---- 最大推力 ----
    max_thrusts = max_rot_vel ** 2 * force_constants  # ≈ [6.0, 6.0, 6.0, 6.0]

    # ---- 混合矩阵 ----
    mixer = _compute_mixer(
        rotor_angles, arm_lengths,
        force_constants, moment_constants,
        directions, inertia_diag,
    )

    return LeeControllerParams(
        pos_gain=position_gain,
        vel_gain=velocity_gain,
        attitude_gain=attitude_gain,
        ang_rate_gain=ang_rate_gain,
        mixer=mixer,
        max_thrusts=max_thrusts,
        mass=jnp.array(0.716),
        gravity=jnp.array([0.0, 0.0, 9.81]),
    )


# ============================================================
# 向量归一化
# ============================================================

def _normalize(v: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """归一化向量，带 eps 防止除零。"""
    return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + eps)


# ============================================================
# 核心控制函数
# ============================================================

def lee_position_control(
    params: LeeControllerParams,
    root_state: jnp.ndarray,
    target_vel: jnp.ndarray,
    target_yaw: jnp.ndarray = None,
) -> jnp.ndarray:
    """
    Lee 位置控制器核心计算：速度指令 → 归一化电机指令 [-1, 1]。

    严格对应 lee_position_controller.py:_compute() 的每一步。

    在速度跟踪模式下（原 VelController 用法）：
      target_pos = 当前位置（位置误差=0），仅跟踪速度

    Args:
        params:      LeeControllerParams 预计算常数
        root_state:  (13,) [pos(3), quat(4), linvel(3), angvel(3)]
        target_vel:  (3,) 目标速度（世界坐标系）
        target_yaw:  (1,) 目标偏航角（弧度），None 则保持当前偏航

    Returns:
        (4,) 归一化电机指令 ∈ [-1, 1]
    """
    cmd, _ = _lee_position_control_core(params, root_state, target_vel, target_yaw)
    return cmd


def lee_position_control_debug(
    params: LeeControllerParams,
    root_state: jnp.ndarray,
    target_vel: jnp.ndarray,
    target_yaw: jnp.ndarray = None,
) -> tuple[jnp.ndarray, LeeDebugInfo]:
    """返回归一化电机指令及关键中间量（用于调试）。"""
    return _lee_position_control_core(params, root_state, target_vel, target_yaw)


def _lee_position_control_core(
    params: LeeControllerParams,
    root_state: jnp.ndarray,
    target_vel: jnp.ndarray,
    target_yaw: jnp.ndarray = None,
) -> tuple[jnp.ndarray, LeeDebugInfo]:
    # ---- Step 1: 解析状态 ----
    rot = root_state[3:7]    # 四元数 [w, x, y, z]
    vel = root_state[7:10]   # 世界系线速度
    ang_vel = root_state[10:13]  # 世界系角速度

    # ---- Step 2: 世界系角速度 → 机体系 ----
    # 对应原代码第 170-171 行: ang_vel = quat_rotate_inverse(rot, ang_vel)
    ang_vel_body = quat_rotate_inverse(rot, ang_vel)

    # ---- Step 3: 目标偏航角 ----
    # 若 target_yaw=None，从当前四元数提取偏航角
    # 对应原代码第 145 行
    if target_yaw is None:
        target_yaw = quaternion_to_euler(rot)[2:3]  # (1,)

    # ---- Step 4: 误差计算 ----
    # 速度跟踪模式: target_pos = pos → pos_error = 0
    # 对应原代码第 173-174 行
    pos_error = jnp.zeros(3)  # pos - target_pos = 0
    vel_error = vel - target_vel

    # ---- Step 5: 期望加速度 ----
    # 对应原代码第 176-181 行
    # acc = pos_error * pos_gain + vel_error * vel_gain - gravity - target_acc
    # target_acc = 0 在速度跟踪模式下
    acc = (pos_error * params.pos_gain
           + vel_error * params.vel_gain
           - params.gravity)

    # ---- Step 6: 旋转矩阵 ----
    R = quat_to_rotation_matrix(rot)  # (3, 3)

    # ---- Step 7: 期望姿态 R_des ----
    # 对应原代码第 183-194 行
    b1_des = jnp.concatenate([
        jnp.cos(target_yaw),
        jnp.sin(target_yaw),
        jnp.zeros_like(target_yaw),
    ])  # (3,)

    b3_des = -_normalize(acc)
    b2_des = _normalize(jnp.cross(b3_des, b1_des))
    R_des = jnp.stack([
        jnp.cross(b2_des, b3_des),
        b2_des,
        b3_des,
    ], axis=-1)  # (3, 3) — 列向量排列

    # ---- Step 8: SO(3) 姿态误差 ----
    # 对应原代码第 195-203 行
    # ang_error_matrix = 0.5 * (R_des^T @ R - R^T @ R_des)
    ang_error_matrix = 0.5 * (R_des.T @ R - R.T @ R_des)  # (3, 3)

    # vee map: 提取反对称矩阵的向量部分
    ang_error = jnp.array([
        ang_error_matrix[2, 1],
        ang_error_matrix[0, 2],
        ang_error_matrix[1, 0],
    ])  # (3,)

    # ---- Step 9: 角加速度 ----
    # 对应原代码第 204-209 行
    # ang_rate_err = ang_vel_body (因为目标角速度=0)
    # 注：忽略欧拉方程耦合项 cross(I*ω, ω)（低速速度控制场景可忽略）
    ang_acc = (
        -ang_error * params.attitude_gain
        - ang_vel_body * params.ang_rate_gain
    )

    # ---- Step 10: 总推力 ----
    # 对应原代码第 210 行
    # thrust = -mass * (acc · R[:, 2])
    thrust = -params.mass * jnp.dot(acc, R[:, 2])

    # ---- Step 11: 混合矩阵 → 电机指令 ----
    # 对应原代码第 211-213 行
    ang_acc_thrust = jnp.concatenate([ang_acc, thrust[None]])  # (4,)
    cmd = params.mixer @ ang_acc_thrust  # (4,)

    # 归一化到 [-1, 1]
    cmd_norm = (cmd / params.max_thrusts) * 2.0 - 1.0

    debug = LeeDebugInfo(
        vel_error=vel_error,
        acc=acc,
        target_yaw=target_yaw,
        ang_vel_body=ang_vel_body,
        ang_error=ang_error,
        ang_acc=ang_acc,
        thrust=thrust[None],
        cmd_norm=cmd_norm,
        ctrl_raw=cmd,
        ctrl_clipped=jnp.clip(cmd, 0.0, params.max_thrusts),
        clip_mask=jnp.logical_or(cmd < 0.0, cmd > params.max_thrusts),
    )

    return cmd_norm, debug


# ============================================================
# MJX 集成接口
# ============================================================

def vel_to_mjx_ctrl(
    params: LeeControllerParams,
    root_state: jnp.ndarray,
    target_vel: jnp.ndarray,
    target_yaw: jnp.ndarray = None,
) -> jnp.ndarray:
    """
    完整管道：速度指令 → MJX 执行器控制量 [0, KF_MAX]。

    对应原代码中 VelController._inv_call() 的完整流程：
      1. 速度指令 → Lee 控制器 → 归一化电机指令 [-1, 1]
      2. 转换为 MJX ctrl: (cmd + 1) / 2 * max_thrust → [0, 6.0] N

    Args:
        params:      LeeControllerParams
        root_state:  (13,) 无人机状态
        target_vel:  (3,) 目标速度（世界坐标系）

    Returns:
        (4,) MJX 执行器控制量 ∈ [0, KF_MAX]
    """
    cmd = lee_position_control(params, root_state, target_vel, target_yaw=target_yaw)

    # 从 [-1, 1] 转换为 [0, max_thrust]
    ctrl = (cmd + 1.0) / 2.0 * params.max_thrusts

    # 裁剪到有效范围
    ctrl = jnp.clip(ctrl, 0.0, params.max_thrusts)

    # NaN 安全（与原代码 torch.nan_to_num_ 对应）
    ctrl = jnp.where(jnp.isnan(ctrl), 0.0, ctrl)

    return ctrl


def vel_to_mjx_ctrl_debug(
    params: LeeControllerParams,
    root_state: jnp.ndarray,
    target_vel: jnp.ndarray,
    target_yaw: jnp.ndarray = None,
) -> tuple[jnp.ndarray, LeeDebugInfo]:
    """返回 MJX 控制量及 Lee 控制器关键中间量（用于排查问题）。"""
    cmd_norm, debug = lee_position_control_debug(
        params,
        root_state,
        target_vel,
        target_yaw=target_yaw,
    )

    ctrl = (cmd_norm + 1.0) / 2.0 * params.max_thrusts
    ctrl = jnp.clip(ctrl, 0.0, params.max_thrusts)
    ctrl = jnp.where(jnp.isnan(ctrl), 0.0, ctrl)

    debug = LeeDebugInfo(
        vel_error=debug.vel_error,
        acc=debug.acc,
        target_yaw=debug.target_yaw,
        ang_vel_body=debug.ang_vel_body,
        ang_error=debug.ang_error,
        ang_acc=debug.ang_acc,
        thrust=debug.thrust,
        cmd_norm=debug.cmd_norm,
        ctrl_raw=debug.ctrl_raw,
        ctrl_clipped=ctrl,
        clip_mask=debug.clip_mask,
    )
    return ctrl, debug
