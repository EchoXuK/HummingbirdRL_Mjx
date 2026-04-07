"""
键盘控制 + 静动态障碍物场景 + 雷达扫描可视化接触点 演示程序。

控制说明：
- W/S: 前进/后退
- A/D: 左移/右移
- Q/E: 左旋/右旋 (Yaw)
- Z/C: 上升/下降
- ESC: 退出
- 空格: 重置飞行器位置
"""

import os
import time
import threading
import mujoco
import mujoco.viewer
import numpy as np
import xml.etree.ElementTree as ET
import jax
import jax.numpy as jnp
from pynput import keyboard

# 导入原生工程依赖
from navigation_env import (
    EnvConfig, 
    _reset_dynamic_obstacles,
    _step_dynamic_obstacles,
    _dynamic_categories_for_count,
    _dynamic_half_extents_from_categories,
    _dynamic_geom_types_from_categories,
)
from envs.obstacle_generator import generate_static_obstacles
from envs.mjcf_scene import build_scene_xml_with_static_obstacles, dynamic_obstacle_body_names

# ==========================================
# 仿真配置
# ==========================================
_TIME_STEP = 0.016  # 与工程默认保持一致
_RAY_VISUAL_SIZE = 0.03  # 雷达红色小球大小 (m)
_LIDAR_YAW_RESOLUTION = 180  # 雷达线数 (180条保证性能和视觉效果)
_LIDAR_RANGE = 4.0  # 雷达最大探测范围

# 定义 Group ID
_STATIC_GROUP = 3
_DYNAMIC_GROUP = 4
_VISUAL_GROUP = 2  # 雷达点所在的 Group

# 解析基础模型路径
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
_XML_PATH = os.path.join(_ASSETS_DIR, "hummingbird.xml")


class KeyboardControl:
    """使用 pynput 在后台线程监听键盘状态"""
    def __init__(self):
        self.cmd_linear_vel = np.zeros(3)  # [vx, vy, vz] 体系下
        self.cmd_yaw_rate = 0.0
        self.reset_requested = False
        self.exit_requested = False
        
        self._linear_gain = 3.0
        self._yaw_rate_gain = np.pi  # 180度/秒
        
        self.pressed_keys = set()
        self.lock = threading.Lock()
        
        # 启动后台监听
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def stop(self):
        self.listener.stop()

    def on_press(self, key):
        with self.lock:
            if key == keyboard.Key.esc:
                self.exit_requested = True
            elif key == keyboard.Key.space:
                self.reset_requested = True
            elif hasattr(key, 'char') and key.char is not None:
                self.pressed_keys.add(key.char.lower())
            self._update_commands()

    def on_release(self, key):
        with self.lock:
            if hasattr(key, 'char') and key.char is not None:
                char = key.char.lower()
                if char in self.pressed_keys:
                    self.pressed_keys.remove(char)
            self._update_commands()

    def _update_commands(self):
        """根据当前按下的按键更新速度指令"""
        self.cmd_linear_vel.fill(0.0)
        self.cmd_yaw_rate = 0.0

        if 'w' in self.pressed_keys: self.cmd_linear_vel[0] = self._linear_gain
        if 's' in self.pressed_keys: self.cmd_linear_vel[0] = -self._linear_gain
        if 'a' in self.pressed_keys: self.cmd_linear_vel[1] = self._linear_gain
        if 'd' in self.pressed_keys: self.cmd_linear_vel[1] = -self._linear_gain
        if 'z' in self.pressed_keys: self.cmd_linear_vel[2] = self._linear_gain
        if 'c' in self.pressed_keys: self.cmd_linear_vel[2] = -self._linear_gain

        if 'q' in self.pressed_keys: self.cmd_yaw_rate = self._yaw_rate_gain
        if 'e' in self.pressed_keys: self.cmd_yaw_rate = -self._yaw_rate_gain


def generate_integrated_scene_xml(seed: int):
    """构建包含静动态障碍物及雷达预分配红点的 MJCF 字符串"""
    print("Generating integrated scene with pre-allocated lidar visualization geoms...")
    config = EnvConfig(
        n_static_obs=150,
        n_dynamic_obs=80,
        physical_static_obstacles=True,
        physical_dynamic_obstacles=True,
        static_obstacle_representation="hfield"
    )
    
    key = jax.random.PRNGKey(seed)
    key, field_key, dyn_key = jax.random.split(key, 3)

    # 1. 生成静态场
    static_field = generate_static_obstacles(
        field_key,
        n_obstacles=config.n_static_obs,
        area_size=config.static_obs_map_range,
        generator_mode=config.static_terrain_generator,
        horizontal_scale=config.static_terrain_hscale,
        vertical_scale=config.static_terrain_vscale,
        platform_width=config.static_terrain_platform_width,
    )

    # 2. 生成动态场初始位姿及几何属性
    categories = _dynamic_categories_for_count(config.n_dynamic_obs)
    half_extents = _dynamic_half_extents_from_categories(categories)
    geom_types = _dynamic_geom_types_from_categories(categories)
    dyn_obs_state = _reset_dynamic_obstacles(dyn_key, config)

    # 3. 组装原生 XML
    mjcf_str = build_scene_xml_with_static_obstacles(
        _XML_PATH,
        static_field,
        area_size=max(config.spawn_distance, config.static_obs_map_range) + config.static_scene_margin,
        terrain_representation=config.static_obstacle_representation,
        hfield_cell_size=config.hfield_cell_size,
        hfield_base_z=config.hfield_base_z,
        dynamic_positions=np.array(dyn_obs_state.positions),
        dynamic_half_extents=np.array(half_extents),
        dynamic_geom_types=geom_types
    )

    # 4. 强行注入雷达可视化点
    root = ET.fromstring(mjcf_str)
    worldbody = root.find("worldbody")
    
    for i in range(_LIDAR_YAW_RESOLUTION):
        name = f"lidar_visual_geom_{i}"
        geom_element = ET.SubElement(worldbody, "geom")
        geom_element.set("name", name)
        geom_element.set("type", "sphere")
        geom_element.set("size", str(_RAY_VISUAL_SIZE))
        geom_element.set("rgba", "1 0.2 0.2 0.8")  # 亮红色
        geom_element.set("group", str(_VISUAL_GROUP))
        geom_element.set("conaffinity", "0")
        geom_element.set("contype", "0")

    return ET.tostring(root, encoding='unicode'), config, dyn_obs_state


def apply_control(mj_data, mj_model, kbd_state):
    """使用直接速度控制覆盖物理状态（简化版交互替代方案）"""
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    vel_offset = mj_model.body_dofadr[body_id]

    with kbd_state.lock:
        cmd_linear_vel = np.copy(kbd_state.cmd_linear_vel)
        cmd_yaw_rate = kbd_state.cmd_yaw_rate

    # 将局部坐标系下设定的速度指令转回世界坐标系
    current_rot_mat = mj_data.xmat[body_id].reshape(3, 3)
    world_vel_cmd = current_rot_mat @ cmd_linear_vel
    
    mj_data.qvel[vel_offset:vel_offset+3] = world_vel_cmd
    mj_data.qvel[vel_offset+3:vel_offset+6] = np.array([0.0, 0.0, cmd_yaw_rate])


def run_viewer():
    kbd = KeyboardControl()
    try:
        seed = int(time.time())
        key = jax.random.PRNGKey(seed)
        
        # 1. 加载场景与配置
        final_xml, config, dyn_obs_state = generate_integrated_scene_xml(seed)
        mj_model = mujoco.MjModel.from_xml_string(final_xml)
        mj_model.opt.timestep = _TIME_STEP
        mj_data = mujoco.MjData(mj_model)

        # 初始化飞行器位置
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        jnt_offset = mj_model.body_jntadr[body_id]
        mj_data.qpos[jnt_offset:jnt_offset+3] = np.array([0.0, 0.0, 2.0]) # 悬停高处
        mujoco.mj_forward(mj_model, mj_data)

        # 动态障碍物 Mocap IDs 缓存
        dynamic_body_names_list = dynamic_obstacle_body_names(config.n_dynamic_obs)
        dynamic_body_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name) for name in dynamic_body_names_list]
        dynamic_mocap_ids = np.array(mj_model.body_mocapid)[np.array(dynamic_body_ids)]

        # 雷达碰撞点 IDs 缓存
        lidar_geom_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"lidar_visual_geom_{i}") for i in range(_LIDAR_YAW_RESOLUTION)]

        dyn_step_count = jnp.array(0, dtype=jnp.int32)
        angles = np.linspace(0, 2 * np.pi, _LIDAR_YAW_RESOLUTION, endpoint=False)
        # [360, 3] 局部坐标系下的水平圆盘射线
        raw_dirs = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=-1)

        print("Launch viewer...")
        with mujoco.viewer.launch_passive(mj_model, mj_data, show_left_ui=False, show_right_ui=False) as viewer:
            # 强制开启隐藏的障碍物与雷达射线球可见性
            viewer.opt.geomgroup[_STATIC_GROUP] = 1
            viewer.opt.geomgroup[_DYNAMIC_GROUP] = 1
            viewer.opt.geomgroup[_VISUAL_GROUP] = 1 

            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = body_id
            viewer.cam.distance = 6.0
            viewer.cam.elevation = -30.0
            
            while viewer.is_running():
                step_start = time.time()
                
                with kbd.lock:
                    if kbd.exit_requested: break
                    if kbd.reset_requested:
                        mj_data.qpos[jnt_offset:jnt_offset+3] = np.array([0.0, 0.0, 2.0])
                        kbd.reset_requested = False

                # --- 1. JAX 动态障碍物步进 ---
                key, step_key = jax.random.split(key)
                dyn_obs_state = _step_dynamic_obstacles(dyn_obs_state, dyn_step_count, step_key, config)
                dyn_step_count += 1
                mj_data.mocap_pos[dynamic_mocap_ids] = np.array(dyn_obs_state.positions)

                # --- 2. 控制指令与物理步进 ---
                apply_control(mj_data, mj_model, kbd)
                mujoco.mj_step(mj_model, mj_data)

                # --- 3. 实时雷达射线投射 (Raycast) ---
                pos = mj_data.xpos[body_id]
                rot = mj_data.xmat[body_id].reshape(3, 3)
                world_dirs = (rot @ raw_dirs.T).T

                # Allocate a buffer to store the hit geometry ID (required by MuJoCo)
                hit_geomid = np.array([-1], dtype=np.int32)
                ray_group_mask = np.array([0, 0, 0, 1, 1, 0], dtype=np.uint8)
                for i in range(_LIDAR_YAW_RESOLUTION):
                    gid = lidar_geom_ids[i]
                    
                    # 传入 ray_group_mask 阻断自我碰撞反馈
                    dist = mujoco.mj_ray(
                        mj_model, 
                        mj_data, 
                        pos, 
                        world_dirs[i], 
                        geomgroup=ray_group_mask, 
                        flg_static=1, 
                        bodyexclude=body_id, 
                        geomid=hit_geomid
                    )
                    
                    if 0 <= dist <= _LIDAR_RANGE:
                        mj_data.geom_xpos[gid] = pos + world_dirs[i] * dist
                    else:
                        mj_data.geom_xpos[gid] = np.array([0, 0, -100]) # 未击中则藏地底

                viewer.sync()
                
                # 锁定帧率
                time_until_next_step = _TIME_STEP - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    finally:
        kbd.stop()
        print("\nExited.")

if __name__ == "__main__":
    run_viewer()