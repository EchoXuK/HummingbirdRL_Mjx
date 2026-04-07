"""
training/train.py
MJX + JAX PPO 训练主入口

迁移自: training/scripts/train.py

使用方式：
  cd /home/xuk/workspace/drone_navrl/mjx_navrl
  python training/train.py
  python training/train.py --num_envs 1024 --total_iterations 5000
  python training/train.py --wandb_mode online
"""

import os
import sys
import time
import argparse
import datetime
import copy
import yaml
import pickle
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from envs.navigation_env import (
    VectorizedNavigationEnv,
    VectorizedFullNavigationEnv,
    EnvConfig,
)
from training.ppo import (
    PPOConfig,
    PPOState,
    create_ppo_state,
    collect_rollout,
    ppo_update,
)
from networks.actor_critic import ActorCritic, ActorCriticFull, beta_mode


# ============================================================
# 配置加载
# ============================================================

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIGS_DIR = os.path.join(_ROOT_DIR, "configs")
_DEFAULT_CONFIG_PATH = os.path.join(_CONFIGS_DIR, "default.yaml")
_ROUTE_PROFILES_PATH = os.path.join(_CONFIGS_DIR, "routes.yaml")
_CHECKPOINT_META_VERSION = 2
_STRUCTURAL_ENV_KEYS = (
    'n_dynamic_obs',
    'n_nearest_dynamic',
    'physical_static_obstacles',
    'physical_dynamic_obstacles',
    'static_obstacle_representation',
    'lidar_use_scene_ray',
    'lidar_scan_dynamic',
)


def _load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def _normalize_route(route: str | None) -> str | None:
    if route is None:
        return None
    route_key = route.strip().lower()
    return route_key or None


def _deep_update_dict(base: dict, overrides: dict) -> dict:
    """递归合并嵌套配置字典。"""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_route_presets() -> dict:
    """从单一真源加载 route/profile 定义。"""
    route_doc = _load_yaml(_ROUTE_PROFILES_PATH)
    return route_doc.get('routes', route_doc)


def apply_route_preset(cfg: dict, route: str | None) -> dict:
    """按正式路线覆盖配置，形成清晰的 8D / full 入口。"""
    route_key = _normalize_route(route)
    if route_key is None:
        return cfg

    route_presets = _load_route_presets()
    if route_key not in route_presets:
        raise ValueError(f"Unsupported route preset: {route}")
    merged = _deep_update_dict(cfg, route_presets[route_key])
    experiment_cfg = copy.deepcopy(merged.get('experiment', {}))
    experiment_cfg['route'] = route_key
    merged['experiment'] = experiment_cfg
    return merged

def load_config(config_path: str = None, route: str | None = None) -> dict:
    """加载配置：default.yaml → 自定义 YAML → route/profile。"""
    cfg = _load_yaml(_DEFAULT_CONFIG_PATH)

    resolved_config_path = config_path or _DEFAULT_CONFIG_PATH
    if os.path.abspath(resolved_config_path) != os.path.abspath(_DEFAULT_CONFIG_PATH):
        cfg = _deep_update_dict(cfg, _load_yaml(resolved_config_path))

    config_route = cfg.get('experiment', {}).get('route')
    final_route = _normalize_route(route or config_route)
    return apply_route_preset(cfg, final_route)


def make_ppo_config(cfg: dict, args: argparse.Namespace = None) -> PPOConfig:
    """从 YAML 配置构建 PPOConfig。"""
    env_cfg = cfg.get('env', {})
    ppo_cfg = cfg.get('ppo', {})
    train_cfg = cfg.get('training', {})

    num_envs = env_cfg.get('num_envs', 2048)
    if args and args.num_envs is not None:
        num_envs = args.num_envs

    use_full_obs = ppo_cfg.get('use_full_obs', False)
    if args and getattr(args, 'use_full_obs', None):
        use_full_obs = True

    return PPOConfig(
        gamma=ppo_cfg.get('gamma', 0.99),
        lmbda=ppo_cfg.get('lmbda', 0.95),
        clip_ratio_actor=ppo_cfg.get('clip_ratio_actor', 0.1),
        clip_ratio_critic=ppo_cfg.get('clip_ratio_critic', 0.1),
        entropy_coeff=ppo_cfg.get('entropy_coeff', 1e-3),
        learning_rate=ppo_cfg.get('learning_rate', 5e-4),
        max_grad_norm=ppo_cfg.get('max_grad_norm', 5.0),
        num_epochs=ppo_cfg.get('num_epochs', 4),
        num_minibatches=ppo_cfg.get('num_minibatches', 16),
        huber_delta=ppo_cfg.get('huber_delta', 10.0),
        action_limit=env_cfg.get('action_limit', 2.0),
        action_dim=3,
        obs_dim=8,
        rollout_length=ppo_cfg.get('rollout_length', 32),
        num_envs=num_envs,
        use_full_obs=use_full_obs,
        n_nearest_dynamic=env_cfg.get('n_nearest_dynamic', 5),
    )


def make_env_config(cfg: dict) -> EnvConfig:
    """从 YAML 配置构建 EnvConfig。"""
    env_cfg = cfg.get('env', {})
    return EnvConfig(
        max_episode_length=env_cfg.get('max_episode_length', 2200),
        action_limit=env_cfg.get('action_limit', 2.0),
        spawn_distance=env_cfg.get('spawn_distance', 24.0),
        spawn_edge_inset=env_cfg.get('spawn_edge_inset', 0.5),
        height_range_min=env_cfg.get('height_range_min', 0.5),
        height_range_max=env_cfg.get('height_range_max', 2.5),
        lidar_range=env_cfg.get('lidar_range', 4.0),
        n_dynamic_obs=env_cfg.get('n_dynamic_obs', 80),
        n_nearest_dynamic=env_cfg.get('n_nearest_dynamic', 5),
        collision_radius=env_cfg.get('collision_radius', 0.3),
        vel_resample_steps=env_cfg.get('vel_resample_steps', 125),
        sim_dt=env_cfg.get('sim_dt', 0.016),
        dyn_obs_vel_min=env_cfg.get('dyn_obs_vel_min', 0.5),
        dyn_obs_vel_max=env_cfg.get('dyn_obs_vel_max', 1.5),
        reward_safety_static_coeff=env_cfg.get('reward_safety_static_coeff', 1.0),
        reward_safety_dynamic_coeff=env_cfg.get('reward_safety_dynamic_coeff', 1.0),
        dyn_obs_local_range=env_cfg.get('dyn_obs_local_range', 5.0),
        dyn_obs_local_range_z=env_cfg.get('dyn_obs_local_range_z', 4.5),
        dyn_obs_map_range=env_cfg.get('dyn_obs_map_range', 20.0),
        regenerate_static_obs_interval=env_cfg.get('regenerate_static_obs_interval', 0),
        physical_static_obstacles=env_cfg.get('physical_static_obstacles', True),
        n_static_obs=env_cfg.get('n_static_obs', 350),
        static_obs_map_range=env_cfg.get('static_obs_map_range', 20.0),
        static_scene_margin=env_cfg.get('static_scene_margin', 1.0),
        static_terrain_generator=env_cfg.get('static_terrain_generator', 'orbit_discrete'),
        static_terrain_hscale=env_cfg.get('static_terrain_hscale', 0.1),
        static_terrain_vscale=env_cfg.get('static_terrain_vscale', 0.1),
        static_terrain_platform_width=env_cfg.get('static_terrain_platform_width', 0.0),
        static_obstacle_representation=env_cfg.get('static_obstacle_representation', 'hfield'),
        hfield_cell_size=env_cfg.get('hfield_cell_size', 0.1),
        hfield_base_z=env_cfg.get('hfield_base_z', 0.1),
        lidar_scan_dynamic=env_cfg.get('lidar_scan_dynamic', False),
        lidar_use_scene_ray=env_cfg.get('lidar_use_scene_ray', True),
        physical_dynamic_obstacles=env_cfg.get('physical_dynamic_obstacles', True),
    )


def checkpoint_uses_full_obs(checkpoint: dict) -> bool:
    """优先从 manifest 读取观测模式，缺失时回退到参数树启发式推断。"""
    manifest = checkpoint.get('manifest', {}) if isinstance(checkpoint, dict) else {}
    ppo_manifest = manifest.get('ppo_config', {}) if isinstance(manifest, dict) else {}
    if 'use_full_obs' in ppo_manifest:
        return bool(ppo_manifest['use_full_obs'])

    obs_mode = manifest.get('obs_mode') if isinstance(manifest, dict) else None
    if obs_mode in ('8d', 'full'):
        return obs_mode == 'full'

    params = checkpoint.get('params', {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(params, dict):
        return False

    param_tree = params.get('params', params)
    if not isinstance(param_tree, dict):
        return False

    return 'LidarCNN_0' in param_tree or 'DynamicObsMLP_0' in param_tree


def build_checkpoint_manifest(
    cfg: dict,
    ppo_config: PPOConfig,
    env_config: EnvConfig,
    *,
    route: str | None,
    iteration: int | None = None,
    total_frames: int | None = None,
    checkpoint_kind: str | None = None,
    scene_version: int | None = None,
) -> dict:
    """构建自描述 checkpoint manifest。"""
    manifest = {
        'meta_version': _CHECKPOINT_META_VERSION,
        'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'route': _normalize_route(route),
        'obs_mode': 'full' if ppo_config.use_full_obs else '8d',
        'checkpoint_kind': checkpoint_kind,
        'iteration': iteration,
        'total_frames': total_frames,
        'scene_version': scene_version,
        'ppo_config': dict(ppo_config._asdict()),
        'env_config': dict(env_config._asdict()),
        'config_snapshot': copy.deepcopy(cfg),
    }
    return manifest


def _validate_checkpoint_compatibility(
    checkpoint: dict,
    ppo_config: PPOConfig,
    env_config: EnvConfig,
) -> None:
    """恢复训练前校验 checkpoint 的结构性兼容性。"""
    manifest = checkpoint.get('manifest', {}) if isinstance(checkpoint, dict) else {}
    if not manifest:
        print("  Warning: 检查点不包含 manifest，将按旧格式兼容加载。")
        return

    ckpt_full_obs = checkpoint_uses_full_obs(checkpoint)
    if ckpt_full_obs != ppo_config.use_full_obs:
        raise ValueError(
            "检查点观测模式与当前配置不一致："
            f"checkpoint={'full' if ckpt_full_obs else '8d'}，"
            f"current={'full' if ppo_config.use_full_obs else '8d'}"
        )

    ckpt_env = manifest.get('env_config', {})
    mismatches = []
    for key in _STRUCTURAL_ENV_KEYS:
        if key in ckpt_env and ckpt_env[key] != getattr(env_config, key):
            mismatches.append((key, ckpt_env[key], getattr(env_config, key)))

    if mismatches:
        mismatch_str = ', '.join(
            f"{key}: checkpoint={old!r} current={new!r}"
            for key, old, new in mismatches
        )
        raise ValueError(f"检查点与当前环境结构不兼容：{mismatch_str}")


def _make_train_step(env, network, optimizer, config: PPOConfig):
    """创建绑定到当前环境场景版本的训练步函数。"""
    @jax.jit
    def _train_step(ppo_state, env_states, obs):
        rollout, new_env_states, final_obs, new_key = collect_rollout(
            ppo_state, env, env_states, obs, network, config,
        )
        ppo_state = ppo_state._replace(key=new_key)
        new_ppo_state, ppo_info = ppo_update(
            ppo_state, rollout, network, optimizer, config,
        )
        rollout_info = {
            'rollout/mean_reward': jnp.mean(rollout['rewards']),
            'rollout/done_rate': jnp.mean(rollout['dones'].astype(jnp.float32)),
            'rollout/rewards':    rollout['rewards'],
            'rollout/dones':      rollout['dones'],
            'rollout/reach_goal': rollout['reach_goal'],
            'rollout/collision':  rollout['collision'],
            'rollout/truncated':  rollout['truncated'],
        }
        return new_ppo_state, new_env_states, final_obs, {**ppo_info, **rollout_info}

    return _train_step


def _build_training_runtime(
    ppo_state: PPOState,
    env,
    network,
    optimizer,
    config: PPOConfig,
):
    """
    为当前场景版本创建训练运行时。

    该函数会：
      1. 重新 reset 批量环境，丢弃旧场景下的 `env_states/obs`
      2. 创建绑定到当前 `env` 的 train/eval 闭包
    """
    key, env_key = jax.random.split(ppo_state.key)
    ppo_state = ppo_state._replace(key=key)
    env_states, obs = jax.jit(env.reset)(env_key)
    train_step_fn = _make_train_step(env, network, optimizer, config)
    eval_rollout_fn = _make_eval_rollout(env, network, config)
    return ppo_state, env_states, obs, train_step_fn, eval_rollout_fn


# ============================================================
# 评估
# ============================================================

def _make_eval_rollout(env, network, config, num_eval_steps=500):
    """
    创建 JIT 编译的评估 rollout 函数（调用一次，复用多次）。

    params 作为显式参数传入（非闭包捕获），避免每次评估重新编译 XLA。
    env / network / config 通过闭包捕获但不变，JAX 可复用编译缓存。
    """
    @jax.jit
    def _eval_rollout(params, env_states, obs, key):
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
            length=num_eval_steps,
        )
        return all_rewards, all_dones  # (num_eval_steps, N)
    return _eval_rollout


def evaluate_policy(
    ppo_state: PPOState,
    env,
    network,
    config: PPOConfig,
    key: jnp.ndarray,
    eval_rollout_fn=None,
    num_eval_steps: int = 500,
) -> dict:
    key, reset_key = jax.random.split(key)
    env_states, obs = jax.jit(env.reset)(reset_key)

    if eval_rollout_fn is None:
        eval_rollout_fn = _make_eval_rollout(env, network, config, num_eval_steps)

    # ---------------------------------------------------------
    # 计时模块 A：纯 GPU/TPU 推理耗时 (包含 500 步的 lax.scan)
    # ---------------------------------------------------------
    t_rollout_start = time.time()
    all_rewards, all_dones = eval_rollout_fn(ppo_state.params, env_states, obs, key)
    
    # 【核心】：强制 CPU 等待 GPU 计算完成，否则计时会失效
    all_rewards.block_until_ready()
    t_rollout_end = time.time()
    rollout_time = t_rollout_end - t_rollout_start

    # ---------------------------------------------------------
    # 计时模块 B：Device -> Host 数据回传与 CPU 统计耗时
    # ---------------------------------------------------------
    t_cpu_start = time.time()
    
    rewards_np = np.array(all_rewards)   # 触发显存到内存的拷贝
    dones_np = np.array(all_dones).astype(bool)

    ep_rewards = np.zeros(config.num_envs, dtype=np.float32)
    ep_lengths = np.zeros(config.num_envs, dtype=np.int32)
    completed_rewards = []
    completed_lengths = []

    for t in range(num_eval_steps):
        ep_rewards += rewards_np[t]
        ep_lengths += 1
        done_mask = dones_np[t]
        n_done = done_mask.sum()
        if n_done > 0:
            completed_rewards.extend(ep_rewards[done_mask].tolist())
            completed_lengths.extend(ep_lengths[done_mask].tolist())
            ep_rewards[done_mask] = 0.0
            ep_lengths[done_mask] = 0

    if len(completed_rewards) > 0:
        mean_reward = float(np.mean(completed_rewards))
        std_reward = float(np.std(completed_rewards))
        mean_ep_len = float(np.mean(completed_lengths))
    else:
        mean_reward = float(np.mean(ep_rewards))
        std_reward = 0.0
        mean_ep_len = float(np.mean(ep_lengths))

    t_cpu_end = time.time()
    cpu_time = t_cpu_end - t_cpu_start

    return {
        'eval/mean_reward': mean_reward,
        'eval/std_reward': std_reward,
        'eval/mean_ep_length': mean_ep_len,
        'eval/episodes_completed': len(completed_rewards),
        # 增加耗时统计字段
        'eval/time_rollout_s': rollout_time,
        'eval/time_cpu_process_s': cpu_time,
        'eval/time_total_s': rollout_time + cpu_time,
    }


# ============================================================
# 保存/加载
# ============================================================

def save_checkpoint(ppo_state: PPOState, path: str, manifest: dict | None = None):
    """保存训练检查点。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 提取可序列化的部分
    checkpoint = {
        'meta_version': _CHECKPOINT_META_VERSION,
        'params': jax.device_get(ppo_state.params),
        'value_norm_state': jax.device_get(ppo_state.value_norm_state),
        'manifest': copy.deepcopy(manifest or {}),
    }
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(path: str) -> dict:
    """加载训练检查点。"""
    with open(path, 'rb') as f:
        return pickle.load(f)


# ============================================================
# 主训练循环
# ============================================================

def train(args):
    """主训练函数。"""
    # 加载配置
    cfg = load_config(args.config, route=getattr(args, 'route', None))
    ppo_config = make_ppo_config(cfg, args)
    env_config = make_env_config(cfg)
    train_cfg = cfg.get('training', {})
    route_name = cfg.get('experiment', {}).get('route')

    total_iterations = train_cfg.get('total_iterations', 18200)
    if args.total_iterations is not None:
        total_iterations = args.total_iterations
    # --max_frames 优先级最高，自动转换为迭代数（对应原代码 max_frame_num: 12e8）
    if getattr(args, 'max_frames', None) is not None:
        frames_per_iter = ppo_config.rollout_length * ppo_config.num_envs
        total_iterations = max(1, args.max_frames // frames_per_iter)
    eval_interval = train_cfg.get('eval_interval', 500)
    save_interval = train_cfg.get('save_interval', 500)
    log_interval = train_cfg.get('log_interval', 10)
    seed = train_cfg.get('seed', 0)
    if args.seed is not None:
        seed = args.seed

    print("=" * 60)
    print("MJX NavRL Training")
    print("=" * 60)
    print(f"  环境数: {ppo_config.num_envs}")
    print(f"  Rollout 长度: {ppo_config.rollout_length}")
    print(f"  总迭代数: {total_iterations}")
    print(f"  学习率: {ppo_config.learning_rate}")
    print(f"  随机种子: {seed}")
    if route_name:
        print(f"  训练路线: {route_name}")
    print(f"  观测模式: {'完整多模态 (LiDAR+障碍物)' if ppo_config.use_full_obs else '简化 8D'}")
    print("=" * 60)

    # Wandb 初始化
    wandb_cfg = cfg.get('wandb', {})
    wandb_mode = wandb_cfg.get('mode', 'disabled')
    if args.wandb_mode is not None:
        wandb_mode = args.wandb_mode

    use_wandb = wandb_mode != 'disabled'
    if use_wandb:
        try:
            import wandb
            run = wandb.init(
                project=wandb_cfg.get('project', 'NavRL-MJX'),
                name=f"{wandb_cfg.get('name', 'train')}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
                config={
                    'ppo': ppo_config._asdict(),
                    'env': env_config._asdict(),
                    'training': train_cfg,
                },
                mode=wandb_mode,
            )
            save_dir = run.dir
        except ImportError:
            print("  Warning: wandb not installed, disabling logging")
            use_wandb = False
            save_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    else:
        save_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')

    # 创建环境（根据模式选择）
    print("\n[1/3] 创建环境...", flush=True)
    if ppo_config.use_full_obs:
        env = VectorizedFullNavigationEnv(ppo_config.num_envs, env_config)
        print("  使用完整多模态环境（含 LiDAR + 静态/动态障碍物）")
    else:
        env = VectorizedNavigationEnv(ppo_config.num_envs, env_config)
        print("  使用简化 8D 观测环境")

    # 创建 PPO
    print("[2/3] 初始化 PPO...", flush=True)
    key = jax.random.PRNGKey(seed)
    ppo_state, network, optimizer = create_ppo_state(key, ppo_config)

    # 从检查点恢复（对应原代码注释掉的 checkpoint 加载逻辑）
    if getattr(args, 'resume_from', None):
        print(f"  从检查点恢复: {args.resume_from}", flush=True)
        ckpt = load_checkpoint(args.resume_from)
        _validate_checkpoint_compatibility(ckpt, ppo_config, env_config)
        ppo_state = ppo_state._replace(
            params=jax.device_put(ckpt['params']),
            value_norm_state=jax.device_put(ckpt['value_norm_state']),
        )
        print("  检查点已加载。")

    # 初始化环境状态与当前场景版本运行时
    print("[3/3] 初始化环境状态...", flush=True)
    ppo_state, env_states, obs, _train_step, _eval_rollout_fn = _build_training_runtime(
        ppo_state, env, network, optimizer, ppo_config,
    )

    # JIT 编译核心函数
    print("\n编译 JIT 函数（首次运行会较慢）...")
    t0 = time.time()
    
    # 1. 预热训练 JIT
    ppo_state, env_states, obs, info = _train_step(ppo_state, env_states, obs)
    jax.tree.map(lambda x: x.block_until_ready(), info)
    print(f"训练 JIT 编译完成 ({time.time() - t0:.1f}s)")

    # 2. 新增：预热评估 JIT
    t1 = time.time()
    dummy_eval_rewards, dummy_eval_dones = _eval_rollout_fn(
        ppo_state.params, env_states, obs, ppo_state.key
    )
    # 强制同步，确保编译完成
    dummy_eval_rewards.block_until_ready()
    print(f"评估 JIT 编译完成 ({time.time() - t1:.1f}s)\n")

    # 训练循环
    print("开始训练...\n")
    total_frames = 0
    best_reward = -float('inf')

    # Episode 级统计追踪（有界环形缓冲）
    # 对应原代码 stats: return, episode_len, reach_goal, collision, truncated
    _EP_BUF_SIZE = 10000
    ep_cur_rewards = np.zeros(ppo_config.num_envs, dtype=np.float32)
    ep_completed_rewards = deque(maxlen=_EP_BUF_SIZE)
    ep_completed_lengths = deque(maxlen=_EP_BUF_SIZE)
    ep_cur_lengths = np.zeros(ppo_config.num_envs, dtype=np.int32)
    ep_completed_reach_goal = deque(maxlen=_EP_BUF_SIZE)
    ep_completed_collision  = deque(maxlen=_EP_BUF_SIZE)
    ep_completed_truncated  = deque(maxlen=_EP_BUF_SIZE)
    total_completed_episodes = 0

    for iteration in range(1, total_iterations + 1):
        t_start = time.time()

        # 训练一步
        ppo_state, env_states, obs, info = _train_step(ppo_state, env_states, obs)

        # 更新 episode 统计（从 rollout rewards/dones/infos 追踪完整 episode）
        rollout_rewards    = np.array(info.pop('rollout/rewards'))     # (T, N)
        rollout_dones      = np.array(info.pop('rollout/dones'))       # (T, N)
        rollout_reach_goal = np.array(info.pop('rollout/reach_goal'))  # (T, N)
        rollout_collision  = np.array(info.pop('rollout/collision'))   # (T, N)
        rollout_truncated  = np.array(info.pop('rollout/truncated'))   # (T, N)
        for t in range(ppo_config.rollout_length):
            ep_cur_rewards += rollout_rewards[t]
            ep_cur_lengths += 1
            done_mask = rollout_dones[t].astype(bool)
            n_done = done_mask.sum()
            if n_done > 0:
                done_rewards = ep_cur_rewards[done_mask]
                done_lengths = ep_cur_lengths[done_mask]
                done_rg  = rollout_reach_goal[t][done_mask].astype(bool)
                done_col = rollout_collision[t][done_mask].astype(bool)
                done_trunc = rollout_truncated[t][done_mask].astype(bool)
                for i in range(n_done):
                    ep_completed_rewards.append(done_rewards[i])
                    ep_completed_lengths.append(done_lengths[i])
                    ep_completed_reach_goal.append(done_rg[i])
                    ep_completed_collision.append(done_col[i])
                    ep_completed_truncated.append(done_trunc[i])
                total_completed_episodes += n_done
                ep_cur_rewards[done_mask] = 0.0
                ep_cur_lengths[done_mask] = 0

        total_frames += ppo_config.rollout_length * ppo_config.num_envs
        t_elapsed = time.time() - t_start
        fps = ppo_config.rollout_length * ppo_config.num_envs / t_elapsed

        # 静态障碍物多样性：定期重新生成（对应原代码每次运行重新生成地形布局）
        # 仅在完整模式下且配置了 regenerate_static_obs_interval > 0 时触发
        regen_interval = env_config.regenerate_static_obs_interval
        if (ppo_config.use_full_obs
                and regen_interval > 0
                and iteration % regen_interval == 0):
            key, regen_key = jax.random.split(ppo_state.key)
            ppo_state = ppo_state._replace(key=key)
            scene_version = env.regenerate_static_obstacles(regen_key)
            ppo_state, env_states, obs, _train_step, _eval_rollout_fn = _build_training_runtime(
                ppo_state, env, network, optimizer, ppo_config,
            )
            print(
                "  静态场景已重生成："
                f"scene_version={scene_version}。已废弃旧环境状态并重建训练/评估 JIT 闭包，"
                "下一次调用将针对新物理场景执行。"
            )

        # 日志
        if iteration % log_interval == 0:
            actor_loss   = float(info['actor_loss'])
            critic_loss  = float(info['critic_loss'])
            entropy      = float(info['entropy'])
            explained_var      = float(info.get('explained_var', float('nan')))
            grad_norm          = float(info.get('grad_norm', float('nan')))
            actor_grad_norm    = float(info.get('actor_grad_norm', float('nan')))
            critic_grad_norm   = float(info.get('critic_grad_norm', float('nan')))
            rollout_mean_rew   = float(info['rollout/mean_reward'])
            rollout_done_rate  = float(info['rollout/done_rate'])
            print(
                f"  iter {iteration:5d} | "
                f"frames {total_frames:10d} | "
                f"fps {fps:7.0f} | "
                f"actor_loss {actor_loss:8.4f} | "
                f"critic_loss {critic_loss:8.4f} | "
                f"entropy {entropy:8.4f} | "
                f"rew/step {rollout_mean_rew:7.4f}"
            )

            # Episode 级统计（有足够样本时）—— 对应原代码 stats: return, episode_len, reach_goal, collision
            if total_completed_episodes >= 10:
                recent_n = 100
                recent_rew  = list(ep_completed_rewards)[-recent_n:]
                recent_len  = list(ep_completed_lengths)[-recent_n:]
                recent_rg   = list(ep_completed_reach_goal)[-recent_n:]
                recent_col  = list(ep_completed_collision)[-recent_n:]
                recent_trunc = list(ep_completed_truncated)[-recent_n:]
                mean_ep_rew  = float(np.mean(recent_rew))
                mean_ep_len  = float(np.mean(recent_len))
                rate_rg      = float(np.mean(recent_rg))
                rate_col     = float(np.mean(recent_col))
                rate_trunc   = float(np.mean(recent_trunc))
                print(
                    f"  {'':5s}   episodes {total_completed_episodes:7d} | "
                    f"mean_ep_rew {mean_ep_rew:7.2f} | "
                    f"mean_ep_len {mean_ep_len:7.1f} | "
                    f"reach_goal {rate_rg:.3f} | "
                    f"collision {rate_col:.3f} | "
                    f"truncated {rate_trunc:.3f}"
                )

            if use_wandb:
                log_info = {
                    'train/actor_loss':        actor_loss,
                    'train/critic_loss':       critic_loss,
                    'train/entropy':           entropy,
                    'train/entropy_loss':      float(info['entropy_loss']),
                    'train/explained_var':     explained_var,
                    'train/grad_norm':         grad_norm,
                    'train/actor_grad_norm':   actor_grad_norm,
                    'train/critic_grad_norm':  critic_grad_norm,
                    'train/fps':               fps,
                    'train/total_frames':      total_frames,
                    'train/rollout_mean_reward': rollout_mean_rew,
                    'train/rollout_done_rate':   rollout_done_rate,
                }
                if total_completed_episodes >= 10:
                    log_info['train/mean_episode_return']  = float(np.mean(list(ep_completed_rewards)[-100:]))
                    log_info['train/mean_episode_length']  = float(np.mean(list(ep_completed_lengths)[-100:]))
                    log_info['train/total_episodes']       = total_completed_episodes
                    log_info['train/reach_goal_rate']      = float(np.mean(list(ep_completed_reach_goal)[-100:]))
                    log_info['train/collision_rate']       = float(np.mean(list(ep_completed_collision)[-100:]))
                    log_info['train/truncated_rate']       = float(np.mean(list(ep_completed_truncated)[-100:]))
                wandb.log(log_info, step=iteration)

        # 评估
        if iteration % eval_interval == 0:
            print(f"\n  [Eval] 迭代 {iteration}...", flush=True)
            
            # 记录整个评估阶段的宏观耗时
            t_eval_macro_start = time.time()
            
            key, eval_key = jax.random.split(ppo_state.key)
            ppo_state = ppo_state._replace(key=key)
            eval_info = evaluate_policy(
                ppo_state, env, network, ppo_config, eval_key,
                eval_rollout_fn=_eval_rollout_fn,
            )
            
            t_eval_macro_end = time.time()
            macro_total_time = t_eval_macro_end - t_eval_macro_start

            print(
                f"  [Eval] mean_reward={eval_info['eval/mean_reward']:.2f} | "
                f"mean_ep_len={eval_info['eval/mean_ep_length']:.1f} | "
                f"episodes={eval_info['eval/episodes_completed']:.0f}"
            )
            
            # 新增：打印精细化的耗时日志
            print(
                f"  [Time] 完整评估总耗时: {macro_total_time:.2f}s "
                f"(纯GPU推理: {eval_info['eval/time_rollout_s']:.2f}s | "
                f"CPU回传与统计: {eval_info['eval/time_cpu_process_s']:.2f}s)"
            )
            if use_wandb:
                wandb.log(eval_info, step=iteration)

            if eval_info['eval/mean_reward'] > best_reward:
                best_reward = eval_info['eval/mean_reward']
                ckpt_path = os.path.join(save_dir, "checkpoint_best.pkl")
                save_checkpoint(
                    ppo_state,
                    ckpt_path,
                    manifest=build_checkpoint_manifest(
                        cfg,
                        ppo_config,
                        env_config,
                        route=route_name,
                        iteration=iteration,
                        total_frames=total_frames,
                        checkpoint_kind='best',
                        scene_version=getattr(env, 'scene_version', None),
                    ),
                )
                print(f"  [Eval] 最佳模型已保存 (reward={best_reward:.2f})")
            print()

        # 定期保存
        if iteration % save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint_{iteration}.pkl")
            save_checkpoint(
                ppo_state,
                ckpt_path,
                manifest=build_checkpoint_manifest(
                    cfg,
                    ppo_config,
                    env_config,
                    route=route_name,
                    iteration=iteration,
                    total_frames=total_frames,
                    checkpoint_kind='periodic',
                    scene_version=getattr(env, 'scene_version', None),
                ),
            )

    # 最终保存
    ckpt_path = os.path.join(save_dir, "checkpoint_final.pkl")
    save_checkpoint(
        ppo_state,
        ckpt_path,
        manifest=build_checkpoint_manifest(
            cfg,
            ppo_config,
            env_config,
            route=route_name,
            iteration=total_iterations,
            total_frames=total_frames,
            checkpoint_kind='final',
            scene_version=getattr(env, 'scene_version', None),
        ),
    )
    print(f"\n训练完成！最终检查点已保存至 {ckpt_path}")
    print(f"总帧数: {total_frames}, 最佳评估奖励: {best_reward:.2f}")

    if use_wandb:
        wandb.finish()


# ============================================================
# 命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="MJX NavRL PPO Training")
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径 (default: configs/default.yaml)')
    parser.add_argument('--route', type=str, default=None, choices=['8d', 'full'],
                        help='正式训练路线预设：8d 或 full（会在加载配置后覆盖关键开关）')
    parser.add_argument('--num_envs', type=int, default=None,
                        help='并行环境数 (override config)')
    parser.add_argument('--total_iterations', type=int, default=None,
                        help='总训练迭代数 (override config)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='总训练帧数（自动转换为迭代数，override total_iterations）')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子 (override config)')
    parser.add_argument('--wandb_mode', type=str, default=None,
                        choices=['online', 'offline', 'disabled'],
                        help='Wandb 模式 (override config)')
    parser.add_argument('--use_full_obs', action='store_true', default=None,
                        help='启用完整多模态观测（含 LiDAR + 障碍物）(override config)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='从检查点文件恢复训练 (例如 checkpoints/checkpoint_500.pkl)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
