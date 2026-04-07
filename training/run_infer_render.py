"""
run_infer_viewer.py
全场景多模态可视化推理程序（支持悬停测试与场景泛化）

功能：
1. 全场景可视化（含动态/静态障碍物、光照、地面）。
2. 连续导航测试：随机目标 -> RL 飞行 -> 到达后悬停 5~10s -> 下一个目标。
3. 碰撞即结束当前回合，共测试 10 回合。
4. 可选开启 `--randomize_scene` 测泛化性（每回合更换地图）。
"""

import os
import sys
import time
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.viewer

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.navigation_env import FullNavigationEnv, EnvConfig, _STATIC_GEOMGROUP_MASK, _DYNAMIC_GEOMGROUP_MASK, _get_obs_full, _extract_root_state
from networks.actor_critic import ActorCriticFull, beta_mode
from training.train import load_config, make_ppo_config, make_env_config, load_checkpoint
from training.eval import _add_target_marker, _render_trajectory

def sync_viewer_full(viewer, viewer_data, model, mjx_data, target_pos, trajectory):
    """
    全场景同步，修复了原 eval.py 中动态障碍物 (mocap) 丢失的问题。
    """
    with viewer.lock() if hasattr(viewer, 'lock') else nullcontext():
        # 1. 同步无人机本体
        viewer_data.qpos[:] = np.array(mjx_data.qpos)
        viewer_data.qvel[:] = np.array(mjx_data.qvel)
        
        # 2. 同步动态障碍物 (Crucial Fix!)
        if hasattr(mjx_data, 'mocap_pos') and mjx_data.mocap_pos.shape[0] > 0:
            viewer_data.mocap_pos[:] = np.array(mjx_data.mocap_pos)
            viewer_data.mocap_quat[:] = np.array(mjx_data.mocap_quat)
            
        # 3. 前向运动学计算更新几何体
        mujoco.mj_forward(model, viewer_data)
        
        # 4. 渲染目标与轨迹
        _add_target_marker(viewer, target_pos)
        _render_trajectory(viewer, trajectory)
    
    if hasattr(viewer, 'sync'):
        viewer.sync()

def sample_valid_target(key_seed, static_field, config: EnvConfig):
    """通过拒绝采样在安全区域生成一个非碰撞目标点"""
    rng = np.random.default_rng(key_seed)
    pos_static = np.array(static_field.positions)
    half_static = np.array(static_field.half_extents)
    
    while True:
        x = rng.uniform(-config.spawn_distance * 0.8, config.spawn_distance * 0.8)
        y = rng.uniform(-config.spawn_distance * 0.8, config.spawn_distance * 0.8)
        z = rng.uniform(config.height_range_min, config.height_range_max)
        pt = np.array([x, y, z])
        
        # 检查是否在静态障碍物内 (加上无人机碰撞半径安全余量)
        diff = np.abs(pos_static - pt[None, :])
        inside = np.all(diff < (half_static + config.collision_radius + 0.3), axis=-1)
        if not np.any(inside):
            return jnp.array(pt, dtype=jnp.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='检查点路径 (best.pkl)')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--randomize_scene', action='store_true', help='开启泛化测试：每回合重新生成不同的障碍物布局')
    args = parser.parse_args()

    print("\n[1/3] 加载配置与检查点...")
    ckpt = load_checkpoint(args.checkpoint)
    params = jax.device_put(ckpt['params'])
    cfg = load_config(args.config, route='full') # 强制使用 full 路线
    ppo_config = make_ppo_config(cfg)
    ppo_config = ppo_config._replace(use_full_obs=True) # 强制全模态
    env_config = make_env_config(cfg)

    print("[2/3] 初始化全场景物理引擎网络...")
    env = FullNavigationEnv(env_config)
    network = ActorCriticFull(action_dim=ppo_config.action_dim)
    
    # ---- 核心 JIT 函数定义 ----
    @jax.jit
    def get_action(params, obs, target_dir):
        """RL 推理获取动作"""
        alpha, beta_param, _ = network.apply(params, obs.state, obs.lidar, obs.direction, obs.dynamic_obs)
        action_norm = beta_mode(alpha, beta_param)
        return 2.0 * action_norm * ppo_config.action_limit - ppo_config.action_limit

    @jax.jit
    def env_step(state, action):
        """环境步进"""
        return env.step(state, action)

    @jax.jit
    def env_reset(key):
        return env.reset(key)

    @jax.jit
    def update_target_state(state, new_target_pos):
        """用于在悬停结束后，为环境注入新目标并重新计算观测"""
        root_state = _extract_root_state(state.mjx_data)
        new_target_dir = new_target_pos - root_state[:3]
        new_state = state._replace(target_pos=new_target_pos, target_dir=new_target_dir)
        new_obs = _get_obs_full(
            root_state, new_target_pos, new_target_dir, new_state.dyn_obs,
            env.static_field.positions, env.static_field.half_extents, env.config, 
            env.mjx_model, new_state.mjx_data, env.base_link_body_id, 
            _STATIC_GEOMGROUP_MASK, _DYNAMIC_GEOMGROUP_MASK, env.config.physical_dynamic_obstacles
        )
        return new_state, new_obs

    print("[3/3] 启动可视化...")
    viewer_data = mujoco.MjData(env.mj_model)
    key = jax.random.PRNGKey(int(time.time()))
    
    with mujoco.viewer.launch_passive(env.mj_model, viewer_data) as viewer:
        # 设置相机追随
        viewer.opt.geomgroup[3] = 1
        viewer.opt.geomgroup[4] = 1
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        viewer.cam.distance = 7.0
        viewer.cam.elevation = -20.0

        for round_idx in range(1, 11):
            if not viewer.is_running(): break
            
            print(f"\n==================== 回合 {round_idx}/10 ====================")
            key, subkey = jax.random.split(key)
            
            # 泛化性测试：每回合换一次地图
            if args.randomize_scene:
                print(" -> [泛化性测试] 正在生成全新的静态障碍物布局...")
                env.regenerate_static_obstacles(subkey)
                # 因为底层 model 变了，需要通知 viewer
                viewer_data = mujoco.MjData(env.mj_model)
                viewer.update_hfield(0) # 更新地形网格

            state, obs = env_reset(subkey)
            trajectory = []
            
            # 每个回合探索多个目标
            targets_reached = 0
            flight_state = "FLYING"
            hover_timer = 0
            hover_duration_steps = int(np.random.uniform(5.0, 10.0) / env.config.sim_dt) # 5~10秒

            target_np = np.array(state.target_pos)
            print(f" -> 任务 {targets_reached+1}: 飞向初始目标 {target_np.round(2)}")

            while viewer.is_running():
                # 1. 状态机控制逻辑
                drone_pos = np.array(state.mjx_data.qpos[:3])
                dist_to_target = np.linalg.norm(target_np - drone_pos)

                if flight_state == "FLYING":
                    action = get_action(params, obs, state.target_dir)
                    if dist_to_target < 0.6: # 接近目标，开始悬停
                        flight_state = "HOVERING"
                        hover_timer = 0
                        hover_duration_steps = int(np.random.uniform(5.0, 10.0) / env.config.sim_dt)
                        print(f" -> 到达目标! 切换至 Lee 控制器悬停 {hover_duration_steps * env.config.sim_dt:.1f} 秒...")
                
                elif flight_state == "HOVERING":
                    # 神奇之处：RL 动作代表目标系速度。直接输入 [0,0,0] 速度，Lee控制器会自动控制平衡实现完美悬停
                    action = jnp.zeros(3) 
                    hover_timer += 1
                    
                    if hover_timer >= hover_duration_steps:
                        targets_reached += 1
                        flight_state = "FLYING"
                        
                        # 生成新目标
                        seed_int = int(time.time() * 1000) % (2**32)
                        new_target_jnp = sample_valid_target(seed_int, env.static_field, env.config)
                        target_np = np.array(new_target_jnp)
                        
                        # JIT 快速更新环境变量
                        state, obs = update_target_state(state, new_target_jnp)
                        print(f" -> 悬停结束。已生成新目标 {target_np.round(2)}，恢复 RL 飞行。")

                # 2. 物理步进
                state, obs, reward, done, info = env_step(state, action)
                
                # 3. 碰撞检测判定 (回合结束)
                if bool(info['collision']):
                    print(" [!] 发生碰撞！当前回合结束。")
                    time.sleep(1) # 停顿一下让人看清碰撞情况
                    break

                # 4. 可视化渲染
                if int(state.step_count) % 5 == 0: # 稍微抽样减小渲染负担
                    trajectory.append(drone_pos)
                    if len(trajectory) > 200: trajectory.pop(0)
                    sync_viewer_full(viewer, viewer_data, env.mj_model, state.mjx_data, target_np, trajectory)
                
                # 控制渲染速度
                time.sleep(env.config.sim_dt * 0.8) # 留一点算力裕度，实现接近 1.0x 真实速率

if __name__ == "__main__":
    main()