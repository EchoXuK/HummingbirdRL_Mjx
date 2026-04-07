"""
utils/math_utils.py
坐标系变换工具函数（JAX版）
迁移自: training/scripts/utils.py  →  vec_to_new_frame / vec_to_world

原理说明：
-----------
原 env.py 中观测量全部在"目标坐标系（goal frame）"中表达：
  - x 轴：对齐初始目标方向（target_dir 在水平面的投影，z=0）
  - y 轴：z_world × x_axis（右手坐标系的横轴）
  - z 轴：x_axis × y_axis（几乎与世界 z 对齐）

这种设计的好处：RL 策略对机头朝向具备旋转不变性，
策略在任意偏航角下行为一致，大幅减少探索难度。

与 PyTorch 版本的映射关系：
  torch.bmm(v, dir.T)       →  jnp.einsum('...i, ...i', v, dir)
  torch.cross(z, x)         →  jnp.cross(z, x)    （广播自动处理）
  tensor.unsqueeze / expand →  jnp.broadcast_to / [..., None]

函数说明：
  vec_to_new_frame(vec, goal_direction):  世界系 → 目标坐标系
  vec_to_world(vec, goal_direction):      目标坐标系 → 世界系
  euler_to_quaternion(rpy):               欧拉角(RPY) → 四元数 [w,x,y,z]
  quaternion_to_euler(q):                 四元数 [w,x,y,z] → 欧拉角(RPY)
  quat_to_rotation_matrix(q):             四元数 → 3×3 旋转矩阵
  quat_rotate(q, v):                      用四元数旋转向量
  quat_rotate_inverse(q, v):              用四元数的逆旋转向量
"""

import jax
import jax.numpy as jnp
from functools import partial


# ============================================================
# 主要坐标变换函数
# ============================================================

def vec_to_new_frame(vec: jnp.ndarray, goal_direction: jnp.ndarray) -> jnp.ndarray:
    """
    将向量从世界坐标系变换到以 goal_direction 为 x 轴的目标坐标系。

    等效于原 utils.py 中的 vec_to_new_frame()。

    目标坐标系定义（与原代码完全一致）：
      x_axis = normalize(goal_direction)        # 对齐目标方向
      y_axis = normalize(z_world × x_axis)      # 水平横轴（右手）
      z_axis = normalize(x_axis × y_axis)       # 接近垂直方向

    Args:
        vec:            (..., 3) 或 (..., N, 3) 待变换向量
        goal_direction: (..., 3) 目标方向（通常是 target_dir 的水平分量，z=0）

    Returns:
        (..., 3) 或 (..., N, 3) 在目标坐标系中的向量分量

    注意：
        支持广播，例如：
          vec (N, K, 3) × goal_direction (N, 3) → 使用 vmap 或手动广播
        对于 env 级别的并行（vmap over envs），每个 env 调用此函数时
        输入是单个环境的数据，无需额外批量维度。
    """
    eps = 1e-7

    # ----- 构建目标坐标系的三个正交轴 -----
    # x 轴：沿目标方向（归一化）
    # 零向量保护：当 norm < 1e-6 时返回恒等坐标系（vec 不变），避免全零轴退化
    norm = jnp.linalg.norm(goal_direction, axis=-1, keepdims=True)
    identity_x = jnp.broadcast_to(jnp.array([1., 0., 0.]), goal_direction.shape)
    x_axis = jnp.where(norm < 1e-6, identity_x,
                       goal_direction / (norm + eps))

    # y 轴：z_world × x_axis（与原代码 torch.cross(z_world, goal_direction_x) 等价）
    # z_world = [0, 0, 1]，cross 展开为：
    #   z × x = (z_y*x_z - z_z*x_y, z_z*x_x - z_x*x_z, z_x*x_y - z_y*x_x)
    #         = (-x_y, x_x, 0)    （因为 z_world = [0, 0, 1]）
    z_world = jnp.array([0., 0., 1.])
    y_axis = jnp.cross(z_world, x_axis)
    y_axis = y_axis / (jnp.linalg.norm(y_axis, axis=-1, keepdims=True) + eps)

    # z 轴：x × y（补全右手系）
    z_axis = jnp.cross(x_axis, y_axis)
    z_axis = z_axis / (jnp.linalg.norm(z_axis, axis=-1, keepdims=True) + eps)

    # ----- 将 vec 投影到新坐标系的三个轴上 -----
    # 等效于旋转矩阵 R = [x_axis; y_axis; z_axis] 与 vec 的矩阵乘法
    # 对于形状 (..., 3)：
    x_new = jnp.sum(vec * x_axis, axis=-1, keepdims=True)  # (..., 1)
    y_new = jnp.sum(vec * y_axis, axis=-1, keepdims=True)  # (..., 1)
    z_new = jnp.sum(vec * z_axis, axis=-1, keepdims=True)  # (..., 1)

    return jnp.concatenate([x_new, y_new, z_new], axis=-1)  # (..., 3)


def vec_to_new_frame_batched(vec: jnp.ndarray, goal_direction: jnp.ndarray) -> jnp.ndarray:
    """
    批量版本：支持 vec (N, K, 3) 和 goal_direction (N, 3) 的混合广播。

    等效于原代码中 len(vec.size()) == 3 的分支。

    Args:
        vec:            (N, K, 3) 批量向量（K 个向量，每个 env 一批）
        goal_direction: (N, 3)    每个 env 对应一个目标方向

    Returns:
        (N, K, 3) 在目标坐标系中的向量
    """
    # goal_direction 扩展到 (N, 1, 3) 以广播到 K 维
    goal_dir_expanded = goal_direction[:, None, :]  # (N, 1, 3)
    return vec_to_new_frame(vec, goal_dir_expanded)


def vec_to_world(vec: jnp.ndarray, goal_direction: jnp.ndarray) -> jnp.ndarray:
    """
    将向量从目标坐标系变换回世界坐标系（逆变换）。

    等效于原 utils.py 中的 vec_to_world()。

    原理：
      world_dir = [1, 0, 0]（世界系 x 轴）
      在 goal 坐标系中，world_dir 的坐标 = vec_to_new_frame([1,0,0], goal_direction)
      由于旋转矩阵是正交矩阵（R^T = R^{-1}），
      世界系向量 = R^T @ goal_系向量 = vec_to_new_frame(vec, world_frame_in_goal)

    Args:
        vec:            (..., 3) 在目标坐标系中的向量
        goal_direction: (..., 3) 目标方向

    Returns:
        (..., 3) 在世界坐标系中的向量
    """
    # world_dir = [1, 0, 0]，广播到与 goal_direction 相同形状
    world_dir = jnp.array([1., 0., 0.])
    # 将世界 x 轴在目标坐标系中的表达（即目标坐标系的基向量在世界系的坐标）
    world_frame_in_goal = vec_to_new_frame(
        jnp.broadcast_to(world_dir, goal_direction.shape),
        goal_direction
    )
    # 利用正交性做逆变换
    return vec_to_new_frame(vec, world_frame_in_goal)


# ============================================================
# 四元数工具函数（用于 _reset_idx 中的姿态初始化）
# ============================================================

def euler_to_quaternion(rpy: jnp.ndarray) -> jnp.ndarray:
    """
    欧拉角（roll, pitch, yaw）转四元数 [w, x, y, z]（内旋 ZYX 顺序）。

    等效于原 utils.py 导入的 euler_to_quaternion。

    Args:
        rpy: (..., 3) [roll, pitch, yaw]（弧度）

    Returns:
        (..., 4) [w, x, y, z] 单位四元数
    """
    roll  = rpy[..., 0:1]
    pitch = rpy[..., 1:2]
    yaw   = rpy[..., 2:3]

    cy = jnp.cos(yaw   * 0.5)
    sy = jnp.sin(yaw   * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cr = jnp.cos(roll  * 0.5)
    sr = jnp.sin(roll  * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return jnp.concatenate([w, x, y, z], axis=-1)  # (..., 4)


def quat_to_rotation_matrix(q: jnp.ndarray) -> jnp.ndarray:
    """
    四元数 [w, x, y, z] 转 3×3 旋转矩阵。

    Args:
        q: (..., 4) [w, x, y, z]

    Returns:
        (..., 3, 3) 旋转矩阵
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # 旋转矩阵元素（来自四元数展开）
    R = jnp.stack([
        jnp.stack([1-2*(y**2+z**2),  2*(x*y-w*z),    2*(x*z+w*y)   ], axis=-1),
        jnp.stack([2*(x*y+w*z),      1-2*(x**2+z**2), 2*(y*z-w*x)  ], axis=-1),
        jnp.stack([2*(x*z-w*y),      2*(y*z+w*x),    1-2*(x**2+y**2)], axis=-1),
    ], axis=-2)  # (..., 3, 3)
    return R


def quat_rotate(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    用四元数 q 旋转向量 v（等效于 R @ v）。

    Args:
        q: (..., 4) [w, x, y, z]
        v: (..., 3)

    Returns:
        (..., 3) 旋转后的向量
    """
    R = quat_to_rotation_matrix(q)
    return jnp.einsum('...ij,...j->...i', R, v)


def quat_axis(q: jnp.ndarray, axis_idx: int = 2) -> jnp.ndarray:
    """
    提取四元数表示的旋转坐标系中的某个轴（0=x, 1=y, 2=z）。

    等效于原 omni_drones.utils.torch.quat_axis 函数。
    在 env.py 中用于获取无人机机体坐标系的轴方向。

    Args:
        q:        (..., 4) [w, x, y, z]
        axis_idx: 轴索引（0, 1, 2）

    Returns:
        (..., 3) 该轴在世界系中的方向向量
    """
    R = quat_to_rotation_matrix(q)
    return R[..., axis_idx]  # 第 axis_idx 列 = 该轴的世界系方向


def quat_rotate_inverse(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    用四元数 q 的逆旋转向量 v（等效于 R^T @ v）。

    迁移自 omni_drones.utils.torch.quat_rotate_inverse。
    Lee 控制器中用于将世界系角速度转换到机体系。

    原理：q^{-1} ⊗ v ⊗ q = R^T v
    展开公式（避免构建旋转矩阵，更高效）：
      result = v * (2w² - 1) - 2w * (q_vec × v) + 2 * q_vec * (q_vec · v)

    Args:
        q: (..., 4) [w, x, y, z]
        v: (..., 3)

    Returns:
        (..., 3) 逆旋转后的向量
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w ** 2 - 1.0)[..., None]
    b = jnp.cross(q_vec, v) * q_w[..., None] * 2.0
    c = q_vec * jnp.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    return a - b + c


def quaternion_to_euler(q: jnp.ndarray) -> jnp.ndarray:
    """
    四元数 [w, x, y, z] 转欧拉角 [roll, pitch, yaw]（弧度）。

    迁移自 omni_drones.utils.torch.quaternion_to_euler。
    Lee 控制器中用于在 target_yaw=None 时提取当前偏航角。

    Args:
        q: (..., 4) [w, x, y, z]

    Returns:
        (..., 3) [roll, pitch, yaw]（弧度）
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = jnp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = jnp.arcsin(jnp.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return jnp.stack([roll, pitch, yaw], axis=-1)


# ============================================================
# vmap 友好的批量接口（用于并行 env）
# ============================================================

# 对单个 env 调用的函数，可直接用 jax.vmap 批量化
# 例：batch_vec_to_new_frame = jax.vmap(vec_to_new_frame)

def build_goal_frame(goal_direction: jnp.ndarray):
    """
    从 goal_direction 构建完整的目标坐标系基向量。

    与 vec_to_new_frame() 中的坐标系构建逻辑一致（含零向量保护）。

    Args:
        goal_direction: (..., 3) 目标方向向量

    Returns:
        tuple of (..., 3) arrays: (x_axis, y_axis, z_axis)
    """
    eps = 1e-7
    norm = jnp.linalg.norm(goal_direction, axis=-1, keepdims=True)
    identity_x = jnp.broadcast_to(jnp.array([1., 0., 0.]), goal_direction.shape)
    x_axis = jnp.where(norm < 1e-6, identity_x,
                       goal_direction / (norm + eps))
    z_world = jnp.array([0., 0., 1.])
    y_axis = jnp.cross(z_world, x_axis)
    y_axis = y_axis / (jnp.linalg.norm(y_axis, axis=-1, keepdims=True) + eps)
    z_axis = jnp.cross(x_axis, y_axis)
    z_axis = z_axis / (jnp.linalg.norm(z_axis, axis=-1, keepdims=True) + eps)
    return x_axis, y_axis, z_axis


def project_to_goal_frame(
    vec: jnp.ndarray,
    x_axis: jnp.ndarray,
    y_axis: jnp.ndarray,
    z_axis: jnp.ndarray,
) -> jnp.ndarray:
    """
    将向量投影到预计算的目标坐标系基向量上。

    等效于 vec_to_new_frame()，但跳过坐标系构建（使用预计算的轴），
    适合同一 goal_direction 需要多次投影的场景。

    Args:
        vec:    (..., 3) 待投影向量
        x_axis: (..., 3) 目标坐标系 x 轴
        y_axis: (..., 3) 目标坐标系 y 轴
        z_axis: (..., 3) 目标坐标系 z 轴

    Returns:
        (..., 3) 在目标坐标系中的向量分量
    """
    x_new = jnp.sum(vec * x_axis, axis=-1, keepdims=True)
    y_new = jnp.sum(vec * y_axis, axis=-1, keepdims=True)
    z_new = jnp.sum(vec * z_axis, axis=-1, keepdims=True)
    return jnp.concatenate([x_new, y_new, z_new], axis=-1)
