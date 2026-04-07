"""
networks/actor_critic.py
Flax ActorCritic 网络（Beta 分布 Actor + Value Critic）

迁移自: training/scripts/ppo.py → PPO 中的网络部分
        training/scripts/utils.py → BetaActor, make_mlp

架构（无 LiDAR CNN 和动态障碍物 MLP）：
  obs (8D)
    → MLP [256, 256] with LeakyReLU + LayerNorm (对应 make_mlp)
    → Actor: 两个 Dense heads → Softplus + 1 + 1e-6 → alpha, beta (各 3D)
    → Critic: Dense → scalar value

Beta 分布：
  动作 ∈ [0, 1]，后续在 PPO 中缩放到 [-action_limit, +action_limit]
  α = 1 + Softplus(linear(features)) + 1e-6
  β = 1 + Softplus(linear(features)) + 1e-6

权重初始化：
  - MLP: Flax 默认 (lecun_normal)
  - Actor heads: orthogonal(0.01)（对应原代码 init_ 函数）
  - Critic head: orthogonal(0.01)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple


# ============================================================
# Beta 分布工具函数（纯 JAX）
# ============================================================

def beta_sample(
    key: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
) -> jnp.ndarray:
    """
    从 Beta(α, β) 分布采样（通过 Gamma 分布技巧）。

    x = g_a / (g_a + g_b), 其中 g_a ~ Gamma(α), g_b ~ Gamma(β)

    Args:
        key:   JAX PRNG key
        alpha: (..., D) α 参数 (> 0)
        beta:  (..., D) β 参数 (> 0)

    Returns:
        (..., D) 采样值 ∈ (0, 1)
    """
    k1, k2 = jax.random.split(key)
    g_a = jax.random.gamma(k1, alpha)
    g_b = jax.random.gamma(k2, beta)
    # 避免 0/0
    sample = g_a / (g_a + g_b + 1e-8)
    # 裁剪到 (eps, 1-eps) 以避免 log(0)
    return jnp.clip(sample, 1e-6, 1.0 - 1e-6)


def beta_log_prob(
    x: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
) -> jnp.ndarray:
    """
    计算 Beta 分布的对数概率密度。

    log p(x; α, β) = (α-1)log(x) + (β-1)log(1-x) - log B(α, β)

    返回独立各维度的对数概率之和（对应 IndependentBeta）。

    Args:
        x:     (..., D) 值 ∈ (0, 1)
        alpha: (..., D)
        beta:  (..., D)

    Returns:
        (...) 对数概率（各维度求和）
    """
    x = jnp.clip(x, 1e-6, 1.0 - 1e-6)
    log_prob = (
        (alpha - 1.0) * jnp.log(x)
        + (beta - 1.0) * jnp.log(1.0 - x)
        - jax.lax.lgamma(alpha)
        - jax.lax.lgamma(beta)
        + jax.lax.lgamma(alpha + beta)
    )
    # 各维度独立，求和（对应 IndependentBeta reinterpreted_batch_ndims=1）
    return jnp.sum(log_prob, axis=-1)


def beta_entropy(
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
) -> jnp.ndarray:
    """
    计算 Beta 分布的熵。

    H = log B(α,β) - (α-1)ψ(α) - (β-1)ψ(β) + (α+β-2)ψ(α+β)

    返回独立各维度的熵之和。

    Args:
        alpha: (..., D)
        beta:  (..., D)

    Returns:
        (...) 熵（各维度求和）
    """
    ab = alpha + beta
    entropy = (
        jax.lax.lgamma(alpha) + jax.lax.lgamma(beta) - jax.lax.lgamma(ab)
        - (alpha - 1.0) * jax.lax.digamma(alpha)
        - (beta - 1.0) * jax.lax.digamma(beta)
        + (ab - 2.0) * jax.lax.digamma(ab)
    )
    return jnp.sum(entropy, axis=-1)


def beta_mode(
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
) -> jnp.ndarray:
    """
    Beta 分布的众数（确定性动作，用于 eval）。

    mode = (α - 1) / (α + β - 2), 当 α,β > 1

    Args:
        alpha: (..., D)
        beta:  (..., D)

    Returns:
        (..., D) 众数 ∈ (0, 1)
    """
    mode = (alpha - 1.0) / (alpha + beta - 2.0 + 1e-8)
    return jnp.clip(mode, 1e-6, 1.0 - 1e-6)


# ============================================================
# Flax ActorCritic 网络
# ============================================================

class ActorCritic(nn.Module):
    """
    Actor-Critic 网络。

    架构：
      obs → MLP(256, 256, LeakyReLU, LayerNorm) → shared features
        → Actor: alpha_head, beta_head → Alpha, Beta parameters
        → Critic: value_head → scalar V(s)

    Attributes:
        action_dim: 动作空间维度（3）
        hidden_dims: MLP 隐藏层维度
    """
    action_dim: int = 3
    hidden_dims: Tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        """
        前向传播。

        Args:
            obs: (..., obs_dim) 观测

        Returns:
            alpha:  (..., action_dim) Beta 分布 α 参数
            beta:   (..., action_dim) Beta 分布 β 参数
            value:  (..., 1) 状态价值
        """
        # ---- 共享 MLP（对应 make_mlp([256, 256])）----
        x = obs
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.leaky_relu(x)
            x = nn.LayerNorm()(x)

        # ---- Actor: Beta 分布参数 ----
        # 对应 BetaActor: alpha = 1 + Softplus(linear) + 1e-6
        # 显式命名：actor_critic.py 中的 Actor/Critic 头层名，供 ppo.py 中梯度范数计算使用
        alpha_raw = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
            name='alpha_head',
        )(x)
        beta_raw = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
            name='beta_head',
        )(x)

        alpha = 1.0 + nn.softplus(alpha_raw) + 1e-6
        beta = 1.0 + nn.softplus(beta_raw) + 1e-6

        # ---- Critic: 状态价值 ----
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
            name='value_head',
        )(x)

        return alpha, beta, value


# ============================================================
# 完整多模态网络（阶段 6）
# ============================================================

class LidarCNN(nn.Module):
    """
    LiDAR 特征提取 CNN 分支。

    对应原代码 PPO 中的 LiDAR CNN：
      输入: (1, 36, 4) → 转置为 Flax NHWC 格式 (36, 4, 1)
      Conv(1→4, (5,3), SAME) → ELU
      Conv(4→16, (5,3), SAME, stride=(2,1)) → ELU
      Conv(16→16, (5,3), SAME, stride=(2,2)) → ELU
      Flatten → Dense(128) + LayerNorm

    输出: (128,) 特征向量
    """

    @nn.compact
    def __call__(self, lidar: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            lidar: (1, 36, 4) LiDAR 扫描
        Returns:
            (128,) 特征向量
        """
        # (1, 36, 4) → (36, 4, 1) for Flax NHWC
        x = jnp.transpose(lidar, (1, 2, 0))  # (H=36, W=4, C=1)

        x = nn.Conv(features=4, kernel_size=(5, 3), padding='SAME', strides=(1, 1))(x)
        x = nn.elu(x)
        x = nn.Conv(features=16, kernel_size=(5, 3), padding='SAME', strides=(2, 1))(x)
        x = nn.elu(x)
        x = nn.Conv(features=16, kernel_size=(5, 3), padding='SAME', strides=(2, 2))(x)
        x = nn.elu(x)

        x = x.reshape(-1)  # Flatten
        x = nn.Dense(128)(x)
        x = nn.LayerNorm()(x)
        return x


class DynamicObsMLP(nn.Module):
    """
    动态障碍物特征提取 MLP 分支。

    对应原代码 PPO 中的动态障碍物 MLP：
      输入: (5, 10) → Flatten → (50,)
      Dense(128) + LeakyReLU + LayerNorm
      Dense(64)  + LeakyReLU + LayerNorm

    输出: (64,) 特征向量
    """

    @nn.compact
    def __call__(self, dynamic_obs: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            dynamic_obs: (n_nearest, 10) 动态障碍物特征
        Returns:
            (64,) 特征向量
        """
        x = dynamic_obs.reshape(-1)  # (n_nearest*10,) = (50,) for n=5
        x = nn.Dense(128)(x)
        x = nn.leaky_relu(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(64)(x)
        x = nn.leaky_relu(x)
        x = nn.LayerNorm()(x)
        return x


class ActorCriticFull(nn.Module):
    """
    完整多模态 Actor-Critic 网络。

    对应原代码 PPO 中含 LiDAR CNN + 动态障碍物 MLP + 共享 MLP 的完整架构。

    特征拼接：
      LidarCNN(lidar)         → 128D
      state                   →   8D
      DynamicObsMLP(dyn_obs)  →  64D
      ─────────────────────── = 200D

    共享 MLP：Dense(256)+LeakyReLU+LayerNorm × 2 → 256D

    Actor/Critic 头：与 ActorCritic 相同。

    Attributes:
        action_dim: 动作维度（3）
    """
    action_dim: int = 3

    @nn.compact
    def __call__(
        self,
        state: jnp.ndarray,       # (8,) 基础状态观测
        lidar: jnp.ndarray,       # (1, 36, 4) LiDAR 扫描
        direction: jnp.ndarray,   # (3,) 目标方向（保留参数签名以兼容调用，但不输入网络）
        dynamic_obs: jnp.ndarray, # (5, 10) 动态障碍物特征
    ):
        """
        前向传播（多模态输入）。

        注：direction 参数保留以兼容 ObsBundle 的调用接口，但不参与网络计算
        （对应 B 代码中 direction 仅用于 vec_to_world 动作变换，不输入特征提取器）。

        Args:
            state:       (8,)
            lidar:       (1, 36, 4)
            direction:   (3,)   — 不输入网络，仅保留参数签名兼容性
            dynamic_obs: (5, 10)

        Returns:
            alpha:  (action_dim,) Beta α 参数
            beta:   (action_dim,) Beta β 参数
            value:  (1,) 状态价值
        """
        # 各分支特征提取
        lidar_feat = LidarCNN()(lidar)           # (128,)
        dyn_feat   = DynamicObsMLP()(dynamic_obs)  # (64,)

        # 拼接：128 + 8 + 64 = 200D（对应 B 代码 CatTensors: _cnn_feature + state + _dynamic_obstacle_feature）
        # 注：direction 不参与拼接，与原代码行为一致
        feat = jnp.concatenate([lidar_feat, state, dyn_feat])  # (200,)

        # 共享 MLP（与 ActorCritic 相同结构）
        x = feat
        for dim in (256, 256):
            x = nn.Dense(dim)(x)
            x = nn.leaky_relu(x)
            x = nn.LayerNorm()(x)

        # Actor 头（显式命名，与 ActorCritic 保持一致，供 ppo.py 梯度范数计算使用）
        alpha_raw = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
            name='alpha_head',
        )(x)
        beta_raw = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
            name='beta_head',
        )(x)
        alpha = 1.0 + nn.softplus(alpha_raw) + 1e-6
        beta  = 1.0 + nn.softplus(beta_raw)  + 1e-6

        # Critic 头
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
            name='value_head',
        )(x)

        return alpha, beta, value
