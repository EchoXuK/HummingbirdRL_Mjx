"""
envs/mjcf_scene.py
MuJoCo 场景装配工具。

职责：
  - 基于基础 hummingbird.xml 动态注入真实静态障碍物 geom
  - 为 FullNavigationEnv 构建可碰撞的 MuJoCo 物理场景

说明：
  - 静态障碍物仍复用 obstacle_generator.py 中与 B 对齐的离散障碍统计分布；
    但这里会把结果提升为真实 MuJoCo `box` geom，而不是只停留在软件 AABB。
  - 采用 contype/conaffinity 位掩码隔离障碍物之间的无意义静态接触：
      drone hull : contype=1, conaffinity=6
      ground     : contype=2, conaffinity=1
      obstacles  : contype=4, conaffinity=1
    从而保证无人机与地面/障碍物可碰撞，而障碍物之间、地面与障碍物之间不会产生额外接触负担。
"""

from __future__ import annotations

from typing import Iterable
from xml.etree import ElementTree as ET

import numpy as np

from envs.obstacle_generator import StaticObstacleField


_STATIC_OBS_BODY_NAME = "static_obstacles"
_STATIC_OBS_PREFIX = "static_obs_"
_STATIC_HFIELD_NAME = "static_terrain_hfield"
_STATIC_GEOM_GROUP = 3
_DYNAMIC_OBS_PREFIX = "dynamic_obs_"
_DYNAMIC_GEOM_PREFIX = "dynamic_obs_geom_"
_DYNAMIC_GEOM_GROUP = 4


def _vec_to_str(values: Iterable[float]) -> str:
    """将长度为 3 的数值序列编码为 MuJoCo XML 属性字符串。"""
    return " ".join(f"{float(v):.6f}" for v in values)


def _array_to_str(values: np.ndarray) -> str:
    """将任意长度数组编码为 MuJoCo XML 属性字符串。"""
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    return " ".join(f"{float(v):.6f}" for v in flat)


def geomgroup_mask(*groups: int, width: int = 6) -> tuple[int, ...]:
    """构造 MuJoCo geom group 过滤掩码。"""
    mask = [0] * width
    for group in groups:
        if 0 <= int(group) < width:
            mask[int(group)] = 1
    return tuple(mask)


def rasterize_obstacles_to_heightfield(
    static_field: StaticObstacleField,
    *,
    area_size: float,
    cell_size: float = 0.25,
) -> np.ndarray:
    """
    将轴对齐 box 障碍物栅格化为高度图。

    语义：对每个网格单元取落入该单元的障碍物顶面最大值，
    从而把原先的 box/AABB 统计场提升为单张连续 height-field。
    """
    if (static_field.height_map is not None
            and abs(float(static_field.height_map_cell_size) - float(cell_size)) < 1e-6
            and float(area_size) >= float(static_field.height_map_area_size)):
        src = np.asarray(static_field.height_map, dtype=np.float32)
        full_size = float(area_size) * 2.0
        ncols = int(np.ceil(full_size / float(cell_size))) + 1
        nrows = ncols
        height_map = np.zeros((nrows, ncols), dtype=np.float32)
        row_off = (nrows - src.shape[0]) // 2
        col_off = (ncols - src.shape[1]) // 2
        height_map[row_off:row_off + src.shape[0], col_off:col_off + src.shape[1]] = src
        return height_map

    full_size = float(area_size) * 2.0
    ncols = int(np.ceil(full_size / float(cell_size))) + 1
    nrows = ncols
    height_map = np.zeros((nrows, ncols), dtype=np.float32)
    x_centers = np.linspace(-area_size, area_size, ncols, dtype=np.float32)
    y_centers = np.linspace(-area_size, area_size, nrows, dtype=np.float32)

    positions = np.asarray(static_field.positions, dtype=np.float32)
    half_extents = np.asarray(static_field.half_extents, dtype=np.float32)

    def _clip_idx(val: float) -> int:
        return int(np.clip(val, 0, ncols - 1))

    for pos, half in zip(positions, half_extents):
        x0 = float(pos[0] - half[0])
        x1 = float(pos[0] + half[0])
        y0 = float(pos[1] - half[1])
        y1 = float(pos[1] + half[1])
        top = float(pos[2] + half[2])

        ix0 = _clip_idx(np.floor((x0 + area_size) / cell_size) - 1)
        ix1 = _clip_idx(np.ceil((x1 + area_size) / cell_size) + 1)
        iy0 = _clip_idx(np.floor((y0 + area_size) / cell_size) - 1)
        iy1 = _clip_idx(np.ceil((y1 + area_size) / cell_size) + 1)

        xs = x_centers[ix0:ix1 + 1]
        ys = y_centers[iy0:iy1 + 1]
        if xs.size == 0 or ys.size == 0:
            continue

        inside_x = (xs >= x0) & (xs <= x1)
        inside_y = (ys >= y0) & (ys <= y1)
        if not np.any(inside_x) or not np.any(inside_y):
            continue

        mask = np.logical_and.outer(inside_y, inside_x)
        block = height_map[iy0:iy1 + 1, ix0:ix1 + 1]
        np.maximum(block, np.where(mask, top, block), out=block)

    return height_map


def _ensure_asset(root: ET.Element) -> ET.Element:
    """确保根节点存在 <asset> 分组。"""
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    return asset


def _configure_hfield_ground(
    root: ET.Element,
    worldbody: ET.Element,
    static_field: StaticObstacleField,
    *,
    area_size: float,
    cell_size: float,
    base_z: float,
    obstacle_rgba: tuple[float, float, float, float],
) -> None:
    """把基础地面替换为 hfield，并将离散 box 场栅格化写入 asset。"""
    asset = _ensure_asset(root)
    hfield = asset.find(f"hfield[@name='{_STATIC_HFIELD_NAME}']")
    if hfield is None:
        hfield = ET.SubElement(asset, "hfield", name=_STATIC_HFIELD_NAME)

    height_map = rasterize_obstacles_to_heightfield(
        static_field,
        area_size=area_size,
        cell_size=cell_size,
    )
    max_height = float(max(np.max(height_map), 1e-3))
    normalized = np.flipud(height_map / max_height)

    hfield.attrib.update({
        "nrow": str(height_map.shape[0]),
        "ncol": str(height_map.shape[1]),
        "elevation": _array_to_str(normalized),
        "size": _array_to_str(np.array([area_size, area_size, max_height, base_z], dtype=np.float32)),
    })
    hfield.attrib.pop("file", None)
    hfield.attrib.pop("content_type", None)

    ground = worldbody.find("geom[@name='ground']")
    if ground is None:
        ground = ET.SubElement(worldbody, "geom", name="ground")

    ground.attrib.update({
        "type": "hfield",
        "hfield": _STATIC_HFIELD_NAME,
        "pos": "0 0 0",
        "rgba": _vec_to_str(obstacle_rgba),
        "contype": "2",
        "conaffinity": "1",
        "condim": "3",
        "friction": "0.8 0.1 0.1",
        "group": str(_STATIC_GEOM_GROUP),
    })
    ground.attrib.pop("size", None)


def _remove_prefixed_bodies(worldbody: ET.Element, prefix: str) -> None:
    """移除 worldbody 下名称带给定前缀的 body。"""
    to_remove = [
        body for body in worldbody.findall("body")
        if body.get("name", "").startswith(prefix)
    ]
    for body in to_remove:
        worldbody.remove(body)


def _append_dynamic_obstacle_bodies(
    worldbody: ET.Element,
    *,
    positions: np.ndarray,
    half_extents: np.ndarray,
    geom_types: tuple[str, ...],
    obstacle_rgba: tuple[float, float, float, float],
) -> None:
    """向场景注入固定拓扑的 mocap 动态障碍物。"""
    rgba = _vec_to_str(obstacle_rgba)
    _remove_prefixed_bodies(worldbody, _DYNAMIC_OBS_PREFIX)

    for idx, (pos, half, geom_type) in enumerate(zip(positions, half_extents, geom_types)):
        body = ET.SubElement(
            worldbody,
            "body",
            name=f"{_DYNAMIC_OBS_PREFIX}{idx}",
            mocap="true",
            pos=_vec_to_str(pos),
            quat="1 0 0 0",
        )
        if geom_type == "cylinder":
            size = np.array([half[0], half[2]], dtype=np.float32)
        elif geom_type == "box":
            size = np.asarray(half, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported dynamic obstacle geom type: {geom_type}")

        ET.SubElement(
            body,
            "geom",
            name=f"{_DYNAMIC_GEOM_PREFIX}{idx}",
            type=geom_type,
            size=_vec_to_str(size),
            rgba=rgba,
            density="0",
            contype="8",
            conaffinity="1",
            condim="3",
            friction="0.8 0.1 0.1",
            group=str(_DYNAMIC_GEOM_GROUP),
        )


def build_scene_xml_with_static_obstacles(
    base_xml_path: str,
    static_field: StaticObstacleField,
    *,
    area_size: float = 24.0,
    terrain_representation: str = "box",
    hfield_cell_size: float = 0.25,
    hfield_base_z: float = 0.25,
    obstacle_rgba: tuple[float, float, float, float] = (0.55, 0.62, 0.68, 1.0),
    dynamic_positions: np.ndarray | None = None,
    dynamic_half_extents: np.ndarray | None = None,
    dynamic_geom_types: tuple[str, ...] | None = None,
    dynamic_rgba: tuple[float, float, float, float] = (0.82, 0.42, 0.18, 0.55),
) -> str:
    """
    基于基础 MJCF 构建带真实静态障碍物的场景 XML。

    Args:
        base_xml_path: 基础 hummingbird.xml 路径。
        static_field:  由 `generate_static_obstacles()` 生成的 AABB 场。
        obstacle_rgba: 障碍物显示颜色。

    Returns:
        str: 可直接传给 `mujoco.MjModel.from_xml_string()` 的 XML 文本。
    """
    tree = ET.parse(base_xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Invalid MJCF: missing <worldbody>")

    container = worldbody.find(f"body[@name='{_STATIC_OBS_BODY_NAME}']")
    if container is not None:
        worldbody.remove(container)

    terrain_representation = terrain_representation.lower()
    if terrain_representation not in {"box", "hfield"}:
        raise ValueError(f"Unsupported terrain_representation: {terrain_representation}")

    if terrain_representation == "hfield":
        _configure_hfield_ground(
            root,
            worldbody,
            static_field,
            area_size=area_size,
            cell_size=hfield_cell_size,
            base_z=hfield_base_z,
            obstacle_rgba=obstacle_rgba,
        )
    else:
        container = ET.SubElement(worldbody, "body", name=_STATIC_OBS_BODY_NAME)

        positions = np.asarray(static_field.positions, dtype=np.float32)
        half_extents = np.asarray(static_field.half_extents, dtype=np.float32)
        rgba = _vec_to_str(obstacle_rgba)

        for idx, (pos, half) in enumerate(zip(positions, half_extents)):
            ET.SubElement(
                container,
                "geom",
                name=f"{_STATIC_OBS_PREFIX}{idx}",
                type="box",
                pos=_vec_to_str(pos),
                size=_vec_to_str(half),
                rgba=rgba,
                density="0",
                contype="4",
                conaffinity="1",
                condim="3",
                friction="0.8 0.1 0.1",
                group=str(_STATIC_GEOM_GROUP),
            )

    if dynamic_positions is not None or dynamic_half_extents is not None or dynamic_geom_types is not None:
        if dynamic_positions is None or dynamic_half_extents is None or dynamic_geom_types is None:
            raise ValueError("dynamic_positions, dynamic_half_extents and dynamic_geom_types must be provided together")
        dyn_positions = np.asarray(dynamic_positions, dtype=np.float32)
        dyn_half_extents = np.asarray(dynamic_half_extents, dtype=np.float32)
        if dyn_positions.shape != dyn_half_extents.shape:
            raise ValueError("dynamic_positions and dynamic_half_extents must share the same shape")
        if dyn_positions.shape[1] != 3:
            raise ValueError("dynamic obstacle positions must have shape (N, 3)")
        if len(dynamic_geom_types) != dyn_positions.shape[0]:
            raise ValueError("dynamic_geom_types length must equal dynamic obstacle count")
        _append_dynamic_obstacle_bodies(
            worldbody,
            positions=dyn_positions,
            half_extents=dyn_half_extents,
            geom_types=dynamic_geom_types,
            obstacle_rgba=dynamic_rgba,
        )

    return ET.tostring(root, encoding="unicode")


def static_obstacle_geom_names(n_obstacles: int) -> list[str]:
    """返回注入到 MJCF 中的静态障碍物 geom 名称列表。"""
    return [f"{_STATIC_OBS_PREFIX}{i}" for i in range(n_obstacles)]


def dynamic_obstacle_body_names(n_obstacles: int) -> list[str]:
    """返回注入到 MJCF 中的动态障碍物 body 名称列表。"""
    return [f"{_DYNAMIC_OBS_PREFIX}{i}" for i in range(n_obstacles)]


def dynamic_obstacle_geom_names(n_obstacles: int) -> list[str]:
    """返回注入到 MJCF 中的动态障碍物 geom 名称列表。"""
    return [f"{_DYNAMIC_GEOM_PREFIX}{i}" for i in range(n_obstacles)]
