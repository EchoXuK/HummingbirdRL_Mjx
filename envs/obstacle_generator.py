"""
envs/obstacle_generator.py
静态障碍物场生成器

在训练启动时调用一次，生成 N 个随机分布的长方体障碍物，
以 JAX 数组形式存储，供 LiDAR 光线投射和 PPO 训练使用。

对应原代码：
  training/scripts/env.py 第 120-130 行（HfDiscreteObstaclesTerrainCfg）
  third_party/.../hf_terrains.py（离散障碍物高度场生成逻辑）

设计原则：
  - 使用 numpy 进行拒绝采样（运行一次，无需 JIT 兼容）
  - 结果转为 JAX 数组，作为编译时常量传入 LiDAR / env
  - AABB（轴对齐包围盒）表示，与 LiDAR 光线求交兼容
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple


# ============================================================
# 数据结构
# ============================================================

class StaticObstacleField(NamedTuple):
    """
    静态障碍物场（编译时常量，不存入 EnvState）。

    所有障碍物用 AABB（轴对齐包围盒）表示，
    positions 为世界坐标系几何中心，half_extents 为半尺寸。
    底部 z = positions[i, 2] - half_extents[i, 2] = 0（与地面齐平）。
    """
    positions:    jnp.ndarray  # (N, 3) 中心位置 [x, y, z_center]
    half_extents: jnp.ndarray  # (N, 3) AABB 半尺寸 [w/2, l/2, h/2]
    height_map:   jnp.ndarray | None = None  # (H, W) 高度图（米）
    height_map_cell_size: float = 0.1
    height_map_area_size: float = 0.0


def _generate_orbit_style_heightfield(
    rng: np.random.Generator,
    *,
    n_obstacles: int,
    area_size: float,
    width_range: tuple[float, float],
    height_options: tuple,
    height_prob: tuple,
    horizontal_scale: float,
    vertical_scale: float,
    platform_width: float,
    obstacle_spacing_pixels: tuple[int, int],
) -> tuple[np.ndarray, list[tuple[float, float, float]], list[tuple[float, float, float]], np.ndarray]:
    """
    近似复现 ORBIT `discrete_obstacles_terrain()` 的离散 heightfield 逻辑。

    返回：
      - 高度图（米）
      - AABB 中心列表
      - AABB 半尺寸列表
      - 障碍物历史（像素坐标）
    """
    width_pixels = int((area_size * 2.0) / horizontal_scale)
    length_pixels = width_pixels
    obs_width_min = int(width_range[0] / horizontal_scale)
    obs_width_max = int(width_range[1] / horizontal_scale)
    platform_width_px = int(platform_width / horizontal_scale)
    probability_length = len(height_prob)

    obs_width_range = np.arange(obs_width_min, obs_width_max, 4, dtype=np.int32)
    obs_length_range = np.arange(obs_width_min, obs_width_max, 4, dtype=np.int32)
    obs_x_range = np.arange(0, width_pixels, 4, dtype=np.int32)
    obs_y_range = np.arange(0, length_pixels, 4, dtype=np.int32)
    if obs_width_range.size == 0:
        obs_width_range = np.array([max(obs_width_min, 1)], dtype=np.int32)
        obs_length_range = obs_width_range.copy()

    lower_bound_pixels, upper_bound_pixels = obstacle_spacing_pixels
    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.float32)
    obstacles_history: list[tuple[int, int, int, int]] = []
    pos_list: list[tuple[float, float, float]] = []
    half_ext_list: list[tuple[float, float, float]] = []

    def good_distance(x: int, y: int, width: int, length: int) -> bool:
        for xp, yp, wp, lp in obstacles_history:
            dx = abs(xp - x) - width
            dy = abs(yp - y) - length
            if dx < 0 and dy < 0:
                continue
            distance = float(np.sqrt(dx ** 2 + dy ** 2))
            if lower_bound_pixels <= distance <= upper_bound_pixels:
                return False
        return True

    for _ in range(n_obstacles):
        roll = int(rng.choice(probability_length, p=np.asarray(height_prob, dtype=np.float64)))
        height_m = float(rng.uniform(height_options[roll], height_options[roll + 1]))
        height_px = max(int(np.rint(height_m / vertical_scale)), 1)

        attempts = 0
        while attempts < 100000:
            width = int(rng.choice(obs_width_range))
            length = int(rng.choice(obs_length_range))
            x_start = int(rng.choice(obs_x_range))
            y_start = int(rng.choice(obs_y_range))
            if x_start + width > width_pixels:
                x_start = width_pixels - width
            if y_start + length > length_pixels:
                y_start = length_pixels - length
            if not obstacles_history or good_distance(x_start, y_start, width, length):
                break
            attempts += 1

        obstacles_history.append((x_start, y_start, width, length))
        hf_raw[x_start:x_start + width, y_start:y_start + length] = height_px * vertical_scale

        center_x = -area_size + (x_start + width / 2.0) * horizontal_scale
        center_y = -area_size + (y_start + length / 2.0) * horizontal_scale
        center_z = 0.5 * height_px * vertical_scale
        pos_list.append((center_x, center_y, center_z))
        half_ext_list.append((0.5 * width * horizontal_scale,
                              0.5 * length * horizontal_scale,
                              0.5 * height_px * vertical_scale))

    x1 = (width_pixels - platform_width_px) // 2
    x2 = (width_pixels + platform_width_px) // 2
    y1 = (length_pixels - platform_width_px) // 2
    y2 = (length_pixels + platform_width_px) // 2
    if platform_width_px > 0:
        hf_raw[x1:x2, y1:y2] = 0.0

    # ORBIT 原函数内部以 (x, y) 写入 heightfield；
    # MuJoCo hfield / 当前 rasterizer 统一使用 (row=y, col=x) 语义，
    # 因此这里显式转置后再缓存。
    return hf_raw.T.copy(), pos_list, half_ext_list, np.asarray(obstacles_history, dtype=np.int32)


# ============================================================
# 生成函数
# ============================================================

def generate_static_obstacles(
    key,
    n_obstacles: int = 350,
    area_size: float = 24.0,
    width_range: tuple = (0.4, 1.1),
    height_options: tuple = (1.0, 1.5, 2.0, 4.0, 6.0),
    height_prob: tuple = (0.10, 0.15, 0.20, 0.55),
    min_spacing: float = 0.5,
    max_attempts_per_obs: int = 10000,
    spawn_clearance: float = 3.0,
    edge_clearance: float = 0.0,
    generator_mode: str = "orbit_discrete",
    horizontal_scale: float = 0.1,
    vertical_scale: float = 0.1,
    platform_width: float = 0.0,
    obstacle_spacing_pixels: tuple[int, int] = (2, 10),
) -> StaticObstacleField:
    """
    生成随机分布的静态长方体障碍物场。

    高度按概率分段采样（对应原代码 obstacle_height_mode="range"）：
      height_prob[k] = 落在 [height_options[k], height_options[k+1]) 段的概率
    宽度/长度在 width_range 内均匀采样（正方形截面）。
    使用拒绝采样保证相邻障碍物有最小间隙。

    对应原代码参数：
      num_obstacles=350
      obstacle_width_range=(0.4, 1.1)
      obstacle_height_range=[1.0, 1.5, 2.0, 4.0, 6.0]
      obstacle_height_probability=[0.1, 0.15, 0.20, 0.55]

    注：此函数只在训练开始时调用一次，不需要 JIT 兼容。
        使用 numpy 拒绝采样；结果转为 JAX float32 数组。

    Args:
        key:                  JAX PRNG key（用于确定性生成）
        n_obstacles:          障碍物数量
        area_size:            分布区域半径（m），障碍物中心在 [-area_size, area_size]²
        width_range:          宽度范围 [min, max] (m)，正方形截面
        height_options:       高度分段边界（m），长度为 K+1
        height_prob:          每段采样概率，长度为 K，sum=1
        min_spacing:          相邻障碍物外壁最小间隙（m）
        max_attempts_per_obs: 单个障碍物最大尝试次数（超过后放宽间距约束）
        spawn_clearance:      无人机起降区半径（m），该范围内不放置障碍物
        edge_clearance:       与障碍场边界之间保留的最小平坦带（按障碍物外壁计算）
        generator_mode:       `orbit_discrete` 复现 ORBIT heightfield 分布，
                      `aabb_rejection` 保留旧拒绝采样路径
        horizontal_scale:     ORBIT heightfield 水平分辨率（m）
        vertical_scale:       ORBIT heightfield 垂直分辨率（m）
        platform_width:       ORBIT 中央平台宽度（m）
        obstacle_spacing_pixels: ORBIT good-distance 像素间距约束

    Returns:
        StaticObstacleField with shapes (n_obstacles, 3)
    """
    # JAX key → numpy seed（保证确定性）
    seed = int(jax.random.randint(key, (), 0, 2 ** 31 - 1))
    rng = np.random.default_rng(seed)

    h_prob = np.array(height_prob, dtype=float)
    h_prob = h_prob / h_prob.sum()  # 归一化
    n_segments = len(height_options) - 1

    pos_list = []       # [(x, y, z_center), ...]
    half_ext_list = []  # [(hw, hl, hh), ...]

    if generator_mode == "orbit_discrete":
        hf_raw, pos_list, half_ext_list, _ = _generate_orbit_style_heightfield(
            rng,
            n_obstacles=n_obstacles,
            area_size=area_size,
            width_range=width_range,
            height_options=height_options,
            height_prob=height_prob,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            platform_width=platform_width,
            obstacle_spacing_pixels=obstacle_spacing_pixels,
        )

        pos_arr = jnp.array(pos_list, dtype=jnp.float32)
        ext_arr = jnp.array(half_ext_list, dtype=jnp.float32)
        return StaticObstacleField(
            positions=pos_arr,
            half_extents=ext_arr,
            height_map=jnp.array(hf_raw, dtype=jnp.float32),
            height_map_cell_size=float(horizontal_scale),
            height_map_area_size=float(area_size),
        )
    if generator_mode != "aabb_rejection":
        raise ValueError(f"Unsupported generator_mode: {generator_mode}")

    # ---- 阶段1：拒绝采样放置主体障碍物 ----
    placed = 0
    total_attempts = 0

    while placed < n_obstacles:
        total_attempts += 1
        if total_attempts > max_attempts_per_obs * n_obstacles:
            break  # 超出总限额，转入阶段2

        # 随机尺寸（正方形截面）
        w = rng.uniform(width_range[0], width_range[1])
        l = rng.uniform(width_range[0], width_range[1])
        seg = rng.choice(n_segments, p=h_prob)
        h = rng.uniform(height_options[seg], height_options[seg + 1])

        # 随机中心位置（留边距避免超出边界）
        margin = max(w, l) / 2.0
        x = rng.uniform(-area_size + margin, area_size - margin)
        y = rng.uniform(-area_size + margin, area_size - margin)
        hw, hl, hh = w / 2.0, l / 2.0, h / 2.0

        # 跳过起降保护区
        if x ** 2 + y ** 2 < spawn_clearance ** 2:
            continue

        # 跳过边缘平坦带（按障碍物外壁而不是中心点计算）
        if ((area_size - (abs(x) + hw)) < edge_clearance
            or (area_size - (abs(y) + hl)) < edge_clearance):
            continue

        z_center = hh  # 底部贴地

        # 拒绝采样：检查最小间隙
        too_close = False
        for k in range(placed):
            px, py, _ = pos_list[k]
            pw, pl, _ = half_ext_list[k]
            # 两障碍物外壁在 XY 平面的间隙
            dx = max(abs(x - px) - hw - pw, 0.0)
            dy = max(abs(y - py) - hl - pl, 0.0)
            gap = (dx ** 2 + dy ** 2) ** 0.5
            if gap < min_spacing:
                too_close = True
                break

        if not too_close:
            pos_list.append((x, y, z_center))
            half_ext_list.append((hw, hl, hh))
            placed += 1

    # ---- 阶段2：若不足 n_obstacles，放宽间距约束补全 ----
    fallback_attempts = 0
    fallback_limit = max_attempts_per_obs * max(1, n_obstacles - placed)
    while placed < n_obstacles and fallback_attempts < fallback_limit:
        fallback_attempts += 1
        w = rng.uniform(width_range[0], width_range[1])
        l = rng.uniform(width_range[0], width_range[1])
        seg = rng.choice(n_segments, p=h_prob)
        h = rng.uniform(height_options[seg], height_options[seg + 1])
        hw, hl, hh = w / 2.0, l / 2.0, h / 2.0
        x = rng.uniform(-area_size + hw, area_size - hw)
        y = rng.uniform(-area_size + hl, area_size - hl)

        if x ** 2 + y ** 2 < spawn_clearance ** 2:
            continue
        if ((area_size - (abs(x) + hw)) < edge_clearance
                or (area_size - (abs(y) + hl)) < edge_clearance):
            continue

        pos_list.append((x, y, hh))
        half_ext_list.append((hw, hl, hh))
        placed += 1

    while placed < n_obstacles:
        w = rng.uniform(width_range[0], width_range[1])
        l = rng.uniform(width_range[0], width_range[1])
        seg = rng.choice(n_segments, p=h_prob)
        h = rng.uniform(height_options[seg], height_options[seg + 1])
        hw, hl, hh = w / 2.0, l / 2.0, h / 2.0

        # 最后兜底仍保持边界安全：只放在中心保护区外，但允许更密集。
        angle = rng.uniform(0.0, 2.0 * np.pi)
        radius = rng.uniform(spawn_clearance + max(hw, hl), max(area_size - edge_clearance - max(hw, hl), spawn_clearance + max(hw, hl) + 1e-3))
        x = np.clip(radius * np.cos(angle), -area_size + hw, area_size - hw)
        y = np.clip(radius * np.sin(angle), -area_size + hl, area_size - hl)
        pos_list.append((x, y, hh))
        half_ext_list.append((hw, hl, hh))
        placed += 1

    pos_arr = jnp.array(pos_list, dtype=jnp.float32)       # (N, 3)
    ext_arr = jnp.array(half_ext_list, dtype=jnp.float32)  # (N, 3)

    return StaticObstacleField(
        positions=pos_arr,
        half_extents=ext_arr,
        height_map=None,
        height_map_cell_size=float(horizontal_scale),
        height_map_area_size=float(area_size),
    )
