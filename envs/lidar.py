"""
envs/lidar.py
纯 JAX LiDAR 实现（软件光线投射，AABB slab 方法）

对应原代码：
  RayCasterCfg(
    attach_yaw_only=True,
    pattern_cfg=patterns.BpearlPatternCfg(
      horizontal_res=10,                        # 10°间隔，36条
      vertical_ray_angles=linspace(-10, 20, 4)  # [-10°, 0°, 10°, 20°]
    )
  )
  输出 lidar_scan = lidar_range - clamp(distance_to_hit, max=lidar_range)
  形状：(1, 36, 4)

设计原则：
  - 预计算 144 条光线方向（模块级常量，JIT 可复用）
  - attach_yaw_only=True：光线只随偏航角旋转（不受 roll/pitch 影响）
  - AABB slab 方法光线求交：用 jnp.where 代替所有条件分支
  - 完全 jax.vmap / jax.jit 兼容，无 Python 分支
  - 输出格式与原代码一致：大值≈lidar_range = 近距障碍物，小值≈0 = 空旷
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import mujoco.mjx as mjx
import numpy as np
from utils.math_utils import quat_rotate

# ============================================================
# LiDAR 规格常量
# ============================================================

LIDAR_RANGE  = 4.0    # 最大量程 (m)
LIDAR_HBEAMS = 36     # 水平光线数 = int(360 / 10)
LIDAR_VBEAMS = 4      # 垂直光线数
LIDAR_HRES   = 10.0   # 水平分辨率 (°)
LIDAR_VFOV   = (-10.0, 20.0)  # 垂直视角范围 (°)


# ============================================================
# 预计算光线方向（模块级常量）
# ============================================================

def _build_ray_directions() -> jnp.ndarray:
    """
    构建 (144, 3) 预计算光线方向，对应偏航=0 时的机体坐标系。

    水平角：0° = 机体 +x（前方），逆时针排列 0°, 10°, ..., 350°
    垂直角：linspace(-10°, 20°, 4) = [-10°, 0°, 10°, 20°]

    排列顺序与原代码 BpearlPatternCfg 一致：
      外层维度=水平（36），内层维度=垂直（4）
      reshape(144,) → reshape(1, 36, 4)

    Returns:
        (144, 3) float32 单位向量
    """
    h_angles_deg = np.arange(0, 360, LIDAR_HRES)           # (36,)
    v_angles_deg = np.linspace(*LIDAR_VFOV, LIDAR_VBEAMS)  # (4,) = [-10, 0, 10, 20]
    h_rad = np.deg2rad(h_angles_deg)  # (36,)
    v_rad = np.deg2rad(v_angles_deg)  # (4,)

    # 外层=水平，内层=垂直 → reshape(1, 36, 4) 时布局正确
    H, V = np.meshgrid(h_rad, v_rad, indexing='ij')  # (36, 4)

    x = np.cos(V) * np.cos(H)  # (36, 4)
    y = np.cos(V) * np.sin(H)  # (36, 4)
    z = np.sin(V)              # (36, 4)

    dirs = np.stack([x, y, z], axis=-1)  # (36, 4, 3)
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs = dirs / np.maximum(norms, 1e-7)

    return jnp.array(dirs.reshape(LIDAR_HBEAMS * LIDAR_VBEAMS, 3), dtype=jnp.float32)


# 模块加载时预计算一次，被所有 jit 函数共享
_RAY_DIRECTIONS = _build_ray_directions()  # (144, 3)


def _rotate_ray_directions(drone_yaw: jnp.ndarray) -> jnp.ndarray:
  """将预计算光线方向按偏航角旋转到世界坐标系。"""
  cos_y = jnp.cos(drone_yaw)
  sin_y = jnp.sin(drone_yaw)
  rx_rot = _RAY_DIRECTIONS[:, 0] * cos_y - _RAY_DIRECTIONS[:, 1] * sin_y
  ry_rot = _RAY_DIRECTIONS[:, 0] * sin_y + _RAY_DIRECTIONS[:, 1] * cos_y
  return jnp.stack([rx_rot, ry_rot, _RAY_DIRECTIONS[:, 2]], axis=-1)  # (144, 3)

# def _rotate_ray_directions(drone_quat: jnp.ndarray) -> jnp.ndarray:
#     # 将 _RAY_DIRECTIONS (144, 3) 用无人机四元数全姿态旋转
#     return jax.vmap(lambda v: quat_rotate(drone_quat, v))(_RAY_DIRECTIONS)

# ============================================================
# 光线-AABB 求交（slab 方法）
# ============================================================

def _ray_aabb_distance(
    ray_origin: jnp.ndarray,       # (3,)
    ray_dir: jnp.ndarray,          # (3,) 已归一化
    box_center: jnp.ndarray,       # (3,)
    box_half_extent: jnp.ndarray,  # (3,)
    max_dist: float = LIDAR_RANGE,
) -> jnp.ndarray:
    """
    光线与 AABB（轴对齐包围盒）求交，返回第一个有效交点距离。

    使用 slab 方法（三对平行面的参数化求交）。
    全用 jnp.where 代替 if，保证 JIT / vmap 安全。

    命中条件：t_exit >= 0 AND t_exit >= t_enter
    无命中时返回 max_dist（即"无障碍物"）。

    Args:
        ray_origin:     (3,) 光线起点
        ray_dir:        (3,) 光线方向（单位向量）
        box_center:     (3,) AABB 几何中心
        box_half_extent:(3,) AABB 半尺寸
        max_dist:       无命中时返回的默认距离

    Returns:
        () 第一个交点到光线起点的距离，范围 [0, max_dist]
    """
    eps = 1e-7

    # 将光线起点移到 box 坐标系（相对偏移）
    rel = ray_origin - box_center  # (3,)

    # 防止除以零：对接近 0 的分量做微小偏置
    safe_dir = jnp.where(jnp.abs(ray_dir) < eps,
                         jnp.sign(ray_dir + eps) * eps,
                         ray_dir)
    inv_dir = 1.0 / safe_dir  # (3,)

    # 计算三对平行面的参数 t
    t_lo = (-box_half_extent - rel) * inv_dir  # (3,)
    t_hi = ( box_half_extent - rel) * inv_dir  # (3,)

    t_near = jnp.minimum(t_lo, t_hi)  # (3,) 进入各 slab 的 t
    t_far  = jnp.maximum(t_lo, t_hi)  # (3,) 离开各 slab 的 t

    t_enter = jnp.max(t_near)   # 进入盒体的最大 t（最晚进入）
    t_exit  = jnp.min(t_far)    # 离开盒体的最小 t（最早离开）

    # 有效命中：光线穿越 box（t_exit >= t_enter）且交点在前方（t_exit >= 0）
    hit = (t_exit >= 0.0) & (t_exit >= t_enter)

    # 第一个交点：取进入点（若进入点在光线起点后面则取 0）
    t_hit = jnp.maximum(t_enter, 0.0)

    return jnp.where(hit, jnp.minimum(t_hit, max_dist), max_dist)


def _ray_scene_distances(
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    ray_origin: jnp.ndarray,
    ray_dirs: jnp.ndarray,
    max_dist: float,
    *,
    bodyexclude: int | Sequence[int] = -1,
    geomgroup: Sequence[int] = (),
) -> jnp.ndarray:
    """
    使用 MJX 对当前 MuJoCo 场景做真实射线查询。

    返回值统一为距离张量；未命中时返回 `max_dist`，从而与 AABB 回退路径保持一致。
    """
    def _single_ray(ray_dir: jnp.ndarray) -> jnp.ndarray:
        dist, _ = mjx.ray(
            mjx_model,
            mjx_data,
            ray_origin,
            ray_dir,
            geomgroup=geomgroup,
            flg_static=True,
            bodyexclude=bodyexclude,
        )
        return jnp.where(dist < 0.0, max_dist, jnp.minimum(dist, max_dist))

    return jax.vmap(_single_ray)(ray_dirs)  # (144,)


def _ray_dynamic_aabb_distances(
    ray_origin: jnp.ndarray,
    ray_dirs: jnp.ndarray,
    dyn_positions: jnp.ndarray,
    dyn_half_extents: jnp.ndarray,
    max_dist: float,
) -> jnp.ndarray:
    """对动态障碍物保持软件 AABB 求交，并返回每条光线的最近距离。"""
    if dyn_positions.shape[0] == 0:
      return jnp.full((ray_dirs.shape[0],), max_dist, dtype=ray_dirs.dtype)

    def _ray_min_dist(ray_dir: jnp.ndarray) -> jnp.ndarray:
        dists = jax.vmap(
            lambda pos, half: _ray_aabb_distance(ray_origin, ray_dir, pos, half, max_dist)
        )(dyn_positions, dyn_half_extents)  # (N_dyn,)
        return jnp.min(dists)

    return jax.vmap(_ray_min_dist)(ray_dirs)  # (144,)


# ============================================================
# 主 LiDAR 计算函数
# ============================================================

def compute_lidar_scan(
    drone_pos: jnp.ndarray,            # (3,) 无人机世界坐标
    drone_yaw: jnp.ndarray,            # () 偏航角 (rad)
    static_positions: jnp.ndarray,     # (N_s, 3) 静态障碍物中心
    static_half_extents: jnp.ndarray,  # (N_s, 3) 静态障碍物半尺寸
    dyn_positions: jnp.ndarray,        # (N_d, 3) 动态障碍物中心
    dyn_half_extents: jnp.ndarray,     # (N_d, 3) 动态障碍物半尺寸（AABB）
    lidar_range: float = LIDAR_RANGE,
    scan_dynamic: bool = True,
    mjx_model: mjx.Model | None = None,
    mjx_data: mjx.Data | None = None,
    bodyexclude: int | Sequence[int] = -1,
    geomgroup: Sequence[int] = (),
    scene_static_geomgroup: Sequence[int] = (),
    scene_dynamic_geomgroup: Sequence[int] = (),
    scene_dynamic_geoms: bool = False,
) -> jnp.ndarray:
    """
    计算单个环境的 LiDAR 扫描结果（软件光线投射）。

    对应原代码 attach_yaw_only=True：
      光线方向只随偏航角旋转，不受 roll/pitch 影响。

    流程：
      1. 预计算光线方向按偏航角旋转（Z 轴旋转矩阵）
      2. 若提供 `mjx_model/mjx_data`，则对当前 MuJoCo 静态几何做真实 scene raycast
      3. 动态障碍物保持 AABB 软件求交（可选）
      4. 对静态/动态结果取最近距离
      5. 转换为 lidar_scan = lidar_range - distance 格式

    Args:
        drone_pos:           (3,) 无人机世界位置
        drone_yaw:           () 偏航角（rad）
        static_positions:    (N_s, 3)
        static_half_extents: (N_s, 3)
        dyn_positions:       (N_d, 3)
        dyn_half_extents:    (N_d, 3)
        lidar_range:         最大量程 (m)
        scan_dynamic:        True（默认）= LiDAR 同时扫描静态和动态障碍物
                             False = 仅扫描静态障碍物（对应原代码
                             mesh_prim_paths=["/World/ground"] 行为）
        mjx_model/mjx_data:  若提供，则静态场景使用 MJX 对真实 MuJoCo geom 求交；
                 否则回退到 `static_positions/static_half_extents` AABB 近似。
        bodyexclude:         需要在 scene raycast 中排除的 body id（通常为无人机本体）
        geomgroup:           MuJoCo geom 组过滤掩码；空序列表示不过滤
        scene_static_geomgroup:
                 静态场景 raycast 使用的 geom group 过滤掩码。
        scene_dynamic_geomgroup:
                 动态场景 raycast 使用的 geom group 过滤掩码。
        scene_dynamic_geoms: True 表示动态障碍物也已注入 MuJoCo 场景，
                 LiDAR 对动态障碍物优先使用 scene ray 而不是 AABB。

    Returns:
        (1, 36, 4) float32
        值域 [0, lidar_range]
        大值 (≈lidar_range) → 近距障碍物
        小值 (≈0)           → 无障碍物（超出量程）
    """
    # 1. 将预计算光线方向旋转到当前偏航角（绕 Z 轴）
    ray_dirs = _rotate_ray_directions(drone_yaw)  # (144, 3)

    # 2. 静态场景距离：优先使用 MJX 对真实 MuJoCo scene 求交
    if mjx_model is not None and mjx_data is not None:
      static_dists = _ray_scene_distances(
        mjx_model,
        mjx_data,
        drone_pos,
        ray_dirs,
        lidar_range,
        bodyexclude=bodyexclude,
        geomgroup=scene_static_geomgroup if scene_static_geomgroup else geomgroup,
      )
    elif static_positions.shape[0] == 0:
      static_dists = jnp.full((ray_dirs.shape[0],), lidar_range, dtype=ray_dirs.dtype)
    else:
      def _ray_min_static_dist(ray_dir: jnp.ndarray) -> jnp.ndarray:
        dists = jax.vmap(
          lambda pos, half: _ray_aabb_distance(drone_pos, ray_dir, pos, half, lidar_range)
        )(static_positions, static_half_extents)  # (N_static,)
        return jnp.min(dists)

      static_dists = jax.vmap(_ray_min_static_dist)(ray_dirs)  # (144,)

    # 3. 动态障碍物距离（当前仍保持 AABB 近似）
    if scan_dynamic:
      if scene_dynamic_geoms and mjx_model is not None and mjx_data is not None:
        dynamic_dists = _ray_scene_distances(
          mjx_model,
          mjx_data,
          drone_pos,
          ray_dirs,
          lidar_range,
          bodyexclude=bodyexclude,
          geomgroup=scene_dynamic_geomgroup if scene_dynamic_geomgroup else geomgroup,
        )
      else:
        dynamic_dists = _ray_dynamic_aabb_distances(
          drone_pos,
          ray_dirs,
          dyn_positions,
          dyn_half_extents,
          lidar_range,
        )
      min_dists = jnp.minimum(static_dists, dynamic_dists)
    else:
      min_dists = static_dists

    # 4. 转换为 lidar_scan 格式（严格对应原代码）
    lidar_scan = lidar_range - jnp.clip(min_dists, 0.0, lidar_range)  # (144,)

    return lidar_scan.reshape(1, LIDAR_HBEAMS, LIDAR_VBEAMS)  # (1, 36, 4)
