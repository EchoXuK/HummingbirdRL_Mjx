"""
training/ppo.py
纯 JAX PPO 实现（支持 jit，全功能）

迁移自: training/scripts/ppo.py → PPO

设计原则：
  - 全部状态封装在 PPOState NamedTuple 中
  - 使用 Optax 管理优化器状态
  - 使用 jax.lax.scan 进行 rollout 收集和 epoch/minibatch 循环
  - 无 Python for 循环，完全 jit 兼容

超参数（严格匹配原代码）：
  - γ=0.99, λ=0.95
  - ε_actor=0.1, ε_critic=0.1
  - Entropy coeff: 1e-3
  - LR: 5e-4 (Adam), grad clip: 5.0
  - 4 epochs, 16 minibatches
  - Huber loss δ=10 for critic
  - actor_loss × action_dim (=3)

动作管道：
  1. PPO 输出 action_normalized ∈ [0,1] (Beta 分布)
  2. 缩放: action_goal = 2 * action_normalized * action_limit - action_limit → [-2, 2] m/s
  3. Env 处理: vec_to_world → Lee 控制器 → MJX ctrl
"""

import os
import sys
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.core import unfreeze
from typing import NamedTuple, Tuple
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from networks.actor_critic import ActorCritic, ActorCriticFull, beta_sample, beta_log_prob, beta_entropy, beta_mode
from utils.value_norm import ValueNormState, value_norm_init, value_norm_update, value_norm_normalize, value_norm_denormalize
from utils.gae import compute_gae


# ============================================================
# PPO 配置
# ============================================================

class PPOConfig(NamedTuple):
    """PPO 超参数配置。"""
    gamma: float = 0.99
    lmbda: float = 0.95
    clip_ratio_actor: float = 0.1
    clip_ratio_critic: float = 0.1
    entropy_coeff: float = 1e-3
    learning_rate: float = 5e-4
    max_grad_norm: float = 5.0
    num_epochs: int = 4
    num_minibatches: int = 16
    huber_delta: float = 10.0
    action_limit: float = 2.0
    action_dim: int = 3
    obs_dim: int = 8
    rollout_length: int = 32       # T: 每次 rollout 的步数
    num_envs: int = 2048           # N: 并行环境数
    use_full_obs: bool = False     # True = 完整多模态环境，False = 简化环境
    n_nearest_dynamic: int = 5     # 完整模式下动态障碍物观测数量（动 obs 形状 (n,10)）


# ============================================================
# PPO 状态
# ============================================================

class PPOState(NamedTuple):
    """PPO 的完整训练状态。"""
    params: dict                   # Flax 模型参数
    opt_state: optax.OptState      # Optax 优化器状态
    value_norm_state: ValueNormState  # 值归一化状态
    key: jnp.ndarray               # PRNG key


# ============================================================
# 初始化
# ============================================================

def create_ppo_state(
    key: jnp.ndarray,
    config: PPOConfig = PPOConfig(),
) -> Tuple[PPOState, ActorCritic, optax.GradientTransformation]:
    """
    创建 PPO 训练状态。

    Returns:
        (ppo_state, network, optimizer)
        当 config.use_full_obs=True 时，network 为 ActorCriticFull
    """
    key, init_key = jax.random.split(key)

    if config.use_full_obs:
        # 完整多模态网络
        network = ActorCriticFull(action_dim=config.action_dim)
        dummy_state   = jnp.zeros(config.obs_dim)                              # (8,)
        dummy_lidar   = jnp.zeros((1, 36, 4))                                  # (1, 36, 4)
        dummy_dir     = jnp.zeros(3)                                            # (3,)
        dummy_dyn     = jnp.zeros((config.n_nearest_dynamic, 10))              # (5, 10)
        params = network.init(init_key, dummy_state, dummy_lidar, dummy_dir, dummy_dyn)
    else:
        # 简化网络
        network = ActorCritic(action_dim=config.action_dim)
        dummy_obs = jnp.zeros(config.obs_dim)
        params = network.init(init_key, dummy_obs)

    # Convert FrozenDict → regular dict for robust JAX pytree handling.
    # FrozenDict's custom pytree registration can be unreliable under WSL2/certain
    # JAX versions, causing intermittent ScopeParamNotFoundError during jit tracing.
    params = unfreeze(params)

    # Verify parameter structure for full obs mode (fail fast with clear message)
    if config.use_full_obs:
        _p = params.get('params', {})
        assert 'LidarCNN_0' in _p, (
            f"Missing 'LidarCNN_0' in params. Keys found: {list(_p.keys())}"
        )
        assert 'Conv_0' in _p['LidarCNN_0'], (
            f"Missing 'Conv_0' in LidarCNN_0. Keys: {list(_p['LidarCNN_0'].keys())}"
        )
        assert 'kernel' in _p['LidarCNN_0']['Conv_0'], (
            f"Missing 'kernel' in Conv_0. Keys: {list(_p['LidarCNN_0']['Conv_0'].keys())}"
        )

    # 创建优化器（Adam + gradient clipping）
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )
    opt_state = optimizer.init(params)

    # 值归一化
    vn_state = value_norm_init()

    ppo_state = PPOState(
        params=params,
        opt_state=opt_state,
        value_norm_state=vn_state,
        key=key,
    )

    return ppo_state, network, optimizer


# ============================================================
# Rollout 收集
# ============================================================

def collect_rollout(
    ppo_state: PPOState,
    env,            # VectorizedNavigationEnv
    env_states,     # batched EnvState
    init_obs: jnp.ndarray,   # (N, obs_dim) 初始观测 — Bug B2: 避免重算
    network: ActorCritic,
    config: PPOConfig,
) -> Tuple[dict, 'EnvState', jnp.ndarray, jnp.ndarray]:
    """
    收集 T 步 rollout 数据。

    使用 jax.lax.scan 遍历 T 个时间步，每步：
      1. 使用 carry 中的 obs → 网络前向 → 采样动作
      2. 缩放动作 → 环境 step（带自动重置）→ new_obs 存入 carry
      3. 存储 transition

    Args:
        ppo_state:  PPO 训练状态
        env:        VectorizedNavigationEnv
        env_states: 当前批量环境状态
        init_obs:   (N, obs_dim) 初始观测
        network:    ActorCritic 网络实例
        config:     PPO 配置

    Returns:
        (rollout_data, final_env_states, final_obs, final_key)
        rollout_data: dict with keys:
          obs:           (T, N, obs_dim)
          actions:       (T, N, action_dim)  — 归一化动作 [0, 1]
          log_probs:     (T, N)
          values:        (T, N)
          rewards:       (T, N)
          dones:         (T, N)
          next_values:   (T, N)
    """
    T = config.rollout_length

    def _step_fn(carry, _):
        """单步 rollout。"""
        env_states, obs, key = carry  # Bug B2: obs 从 carry 获取，无需重算
        key, action_key, env_key = jax.random.split(key, 3)

        # 网络前向（根据模式选择调用方式）
        # 简化模式：ActorCritic 原生支持批量输入
        # 完整模式：ActorCriticFull 不支持批量输入，使用 vmap
        if config.use_full_obs:
            alpha, beta, value = jax.vmap(
                lambda s, l, d, dy: network.apply(ppo_state.params, s, l, d, dy)
            )(obs.state, obs.lidar, obs.direction, obs.dynamic_obs)
        else:
            alpha, beta, value = network.apply(ppo_state.params, obs)
        # alpha: (N, action_dim), beta: (N, action_dim), value: (N, 1)
        value = value.squeeze(-1)  # (N,)

        # 采样动作（Beta 分布）
        action_normalized = beta_sample(action_key, alpha, beta)  # (N, action_dim) ∈ (0, 1)
        log_prob = beta_log_prob(action_normalized, alpha, beta)  # (N,)

        # 缩放到目标坐标系速度
        action_goal = 2.0 * action_normalized * config.action_limit - config.action_limit  # [-2, 2]

        # 环境步进（带自动重置），new_obs 已包含 autoreset 后的正确观测
        new_env_states, new_obs, rewards, dones, infos = env.step_with_autoreset(
            env_states, action_goal, env_key,
        )

        # 存储 transition（含 episode 统计用的 infos）
        transition = {
            'obs': obs,
            'actions': action_normalized,
            'log_probs': log_prob,
            'values': value,
            'rewards': rewards,
            'dones': dones,
            'reach_goal': infos['reach_goal'],   # (N,) 对应原代码 stats.reach_goal
            'collision':  infos['collision'],    # (N,) 对应原代码 stats.collision
            'truncated':  infos['truncated'],    # (N,) 对应原代码 stats.truncated
        }

        return (new_env_states, new_obs, key), transition  # Bug B2: new_obs 进入 carry

    # Scan over T steps
    (final_env_states, final_obs, final_key), rollout = jax.lax.scan(
        _step_fn,
        (env_states, init_obs, ppo_state.key),  # Bug B2: init_obs 放入初始 carry
        None,
        length=T,
    )
    # rollout: dict of (T, N, ...) arrays
    # final_obs: (N, obs_dim) — 最后一步 step_with_autoreset 返回的观测，无需重算

    # 计算最终 next_values（用于 GAE 的 bootstrap）
    if config.use_full_obs:
        _, _, final_value = jax.vmap(
            lambda s, l, d, dy: network.apply(ppo_state.params, s, l, d, dy)
        )(final_obs.state, final_obs.lidar, final_obs.direction, final_obs.dynamic_obs)
    else:
        _, _, final_value = network.apply(ppo_state.params, final_obs)
    final_value = final_value.squeeze(-1)  # (N,)

    # 构建 next_values: 对于 t < T-1，使用 rollout 中下一步的 value；最后一步用 final_value
    next_values = jnp.concatenate([
        rollout['values'][1:],          # (T-1, N)
        final_value[None],              # (1, N)
    ], axis=0)  # (T, N)

    rollout['next_values'] = next_values

    return rollout, final_env_states, final_obs, final_key


# ============================================================
# Huber Loss
# ============================================================

def huber_loss(pred: jnp.ndarray, target: jnp.ndarray, delta: float = 10.0) -> jnp.ndarray:
    """
    Huber loss（平滑 L1 loss）。

    对应原代码 nn.HuberLoss(delta=10)。

    注意：PyTorch HuberLoss 默认 reduction='mean'。
    """
    diff = pred - target
    abs_diff = jnp.abs(diff)
    loss = jnp.where(
        abs_diff <= delta,
        0.5 * diff ** 2,
        delta * (abs_diff - 0.5 * delta),
    )
    return jnp.mean(loss)


# ============================================================
# PPO 更新
# ============================================================

def ppo_update(
    ppo_state: PPOState,
    rollout: dict,
    network: ActorCritic,
    optimizer: optax.GradientTransformation,
    config: PPOConfig,
) -> Tuple[PPOState, dict]:
    """
    PPO 策略更新。

    对应原代码 PPO.train()。

    流程：
      1. 反归一化 values/next_values
      2. 计算 GAE
      3. 归一化优势
      4. 更新值归一化统计
      5. 归一化回报
      6. 多 epoch × minibatch 梯度更新

    Args:
        ppo_state:  当前 PPO 状态
        rollout:    collect_rollout 返回的 rollout 数据
        network:    ActorCritic 网络实例
        optimizer:  Optax 优化器
        config:     PPO 配置

    Returns:
        (new_ppo_state, info_dict)
    """
    # 解包
    obs = rollout['obs']               # (T, N, obs_dim) 或 ObsBundle
    actions = rollout['actions']       # (T, N, action_dim)
    old_log_probs = rollout['log_probs']  # (T, N)
    values = rollout['values']         # (T, N)
    rewards = rollout['rewards']       # (T, N)
    dones = rollout['dones']           # (T, N)
    next_values = rollout['next_values']  # (T, N)

    # 1. 反归一化 values
    values_denorm = value_norm_denormalize(ppo_state.value_norm_state, values)
    next_values_denorm = value_norm_denormalize(ppo_state.value_norm_state, next_values)

    # 2. 计算 GAE
    advantages, returns = compute_gae(
        rewards, dones, values_denorm, next_values_denorm,
        gamma=config.gamma, lmbda=config.lmbda,
    )

    # 3. 归一化优势
    adv_mean = jnp.mean(advantages)
    adv_std = jnp.std(advantages)
    advantages = (advantages - adv_mean) / jnp.maximum(adv_std, 1e-7)

    # 4. 更新值归一化统计
    new_vn_state = value_norm_update(ppo_state.value_norm_state, returns)

    # 5. 归一化回报
    returns_norm = value_norm_normalize(new_vn_state, returns)

    # 6. 展平时间和环境维度
    T, N = obs.shape[:2] if not config.use_full_obs else obs.state.shape[:2]
    total = T * N

    # jax.tree.map 对 plain array 和 ObsBundle pytree 均适用：
    #   plain array (T, N, d) → (T*N, d)
    #   ObsBundle field (T, N, ...) → (T*N, ...)
    flat_obs = jax.tree.map(lambda x: x.reshape(total, *x.shape[2:]), obs)
    flat_actions = actions.reshape(total, -1)
    flat_old_log_probs = old_log_probs.reshape(total)
    flat_values = values.reshape(total)
    flat_advantages = advantages.reshape(total)
    flat_returns = returns_norm.reshape(total)

    # 7. Epoch × Minibatch 循环
    def _epoch_step(carry, _epoch_key):
        """单个 epoch。"""
        params, opt_state = carry

        # 随机打乱
        perm = jax.random.permutation(_epoch_key, total)
        # 截断到可整除的大小
        usable = (total // config.num_minibatches) * config.num_minibatches
        perm = perm[:usable]
        indices = perm.reshape(config.num_minibatches, -1)

        def _minibatch_step(carry_mb, mb_indices):
            """单个 minibatch 更新。"""
            params_mb, opt_state_mb = carry_mb

            # jax.tree.map 对 plain array 和 ObsBundle 均适用
            mb_obs = jax.tree.map(lambda x: x[mb_indices], flat_obs)
            mb_actions = flat_actions[mb_indices]
            mb_old_log_probs = flat_old_log_probs[mb_indices]
            mb_values = flat_values[mb_indices]
            mb_advantages = flat_advantages[mb_indices]
            mb_returns = flat_returns[mb_indices]

            def loss_fn(p):
                # 网络前向（根据模式选择调用方式）
                if config.use_full_obs:
                    alpha, beta, value = jax.vmap(
                        lambda s, l, d, dy: network.apply(p, s, l, d, dy)
                    )(mb_obs.state, mb_obs.lidar, mb_obs.direction, mb_obs.dynamic_obs)
                else:
                    # Bug B4 优化：ActorCritic 原生支持批量输入
                    alpha, beta, value = network.apply(p, mb_obs)
                value = value.squeeze(-1)

                # Actor loss
                log_probs = beta_log_prob(mb_actions, alpha, beta)
                ratio = jnp.exp(log_probs - mb_old_log_probs)
                # 扩展 advantage 维度以匹配 ratio
                surr1 = mb_advantages * ratio
                surr2 = mb_advantages * jnp.clip(
                    ratio,
                    1.0 - config.clip_ratio_actor,
                    1.0 + config.clip_ratio_actor,
                )
                # actor_loss = -jnp.mean(jnp.minimum(surr1, surr2)) * config.action_dim
                actor_loss = -jnp.mean(jnp.minimum(surr1, surr2)) 
                # Entropy loss
                entropy = beta_entropy(alpha, beta)
                entropy_loss = -config.entropy_coeff * jnp.mean(entropy)

                # Critic loss (clipped Huber)
                value_clipped = mb_values + jnp.clip(
                    value - mb_values,
                    -config.clip_ratio_critic,
                    config.clip_ratio_critic,
                )
                critic_loss_original = huber_loss(value, mb_returns, config.huber_delta)
                critic_loss_clipped = huber_loss(value_clipped, mb_returns, config.huber_delta)
                critic_loss = jnp.maximum(critic_loss_original, critic_loss_clipped)

                total_loss = actor_loss + critic_loss + entropy_loss

                # Explained variance (与原代码 ppo.py:170 对应)
                explained_var = 1.0 - jnp.mean((value - mb_returns) ** 2) / jnp.maximum(jnp.var(mb_returns), 1e-8)

                return total_loss, {
                    'actor_loss': actor_loss,
                    'critic_loss': critic_loss,
                    'entropy_loss': entropy_loss,
                    'entropy': jnp.mean(entropy),
                    'explained_var': explained_var,
                }

            (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_mb)

            # 梯度范数（裁剪前，全局）
            grad_norm = optax.global_norm(grads)

            # Actor/Critic 分组梯度范数（对应原代码 actor_grad_norm, critic_grad_norm）
            # 通过 actor_critic.py 中显式命名的层（alpha_head/beta_head → actor，value_head → critic）
            # 匹配，避免依赖 Flax 自动命名（Dense_2 等）造成的脆弱性。
            # tree_leaves_with_path 在 trace 时静态过滤，JIT 安全
            actor_sq = jnp.array(0.0)
            critic_sq = jnp.array(0.0)
            for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
                key_strs = [
                    k.key if hasattr(k, 'key') else str(k)
                    for k in path
                ]
                if 'alpha_head' in key_strs or 'beta_head' in key_strs:
                    actor_sq = actor_sq + jnp.sum(leaf ** 2)
                elif 'value_head' in key_strs:
                    critic_sq = critic_sq + jnp.sum(leaf ** 2)
            actor_grad_norm  = jnp.sqrt(actor_sq)
            critic_grad_norm = jnp.sqrt(critic_sq)

            updates, new_opt_state = optimizer.update(grads, opt_state_mb, params_mb)
            new_params = optax.apply_updates(params_mb, updates)

            return (new_params, new_opt_state), {
                **info,
                'grad_norm': grad_norm,
                'actor_grad_norm': actor_grad_norm,
                'critic_grad_norm': critic_grad_norm,
            }

        (params, opt_state), epoch_infos = jax.lax.scan(
            _minibatch_step,
            (params, opt_state),
            indices,
        )

        return (params, opt_state), epoch_infos

    # 生成 epoch keys
    key, *epoch_keys = jax.random.split(ppo_state.key, config.num_epochs + 1)
    epoch_keys = jnp.stack(epoch_keys)

    (new_params, new_opt_state), all_infos = jax.lax.scan(
        _epoch_step,
        (ppo_state.params, ppo_state.opt_state),
        epoch_keys,
    )

    # 聚合 info: all_infos 形状为 (num_epochs, num_minibatches)
    info = jax.tree.map(lambda x: jnp.mean(x), all_infos)

    new_ppo_state = PPOState(
        params=new_params,
        opt_state=new_opt_state,
        value_norm_state=new_vn_state,
        key=key,
    )

    return new_ppo_state, info
