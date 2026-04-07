"""
utils/gae.py
广义优势估计（GAE）的 JAX 实现

迁移自: training/scripts/utils.py → GAE

原理（Schulman et al., 2016）：
  δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
  A_t = δ_t + (γλ)(1-done_t) * A_{t+1}

使用 jax.lax.scan 从后往前扫描，避免 Python for 循环，
完全兼容 jit。
"""

import jax
import jax.numpy as jnp
from typing import Tuple


def compute_gae(
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    values: jnp.ndarray,
    next_values: jnp.ndarray,
    gamma: float = 0.99,
    lmbda: float = 0.95,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    计算广义优势估计（GAE）。

    严格对应原代码 GAE.forward()，但使用 jax.lax.scan 实现反向扫描。

    数据布局：
      原代码: (num_envs, num_steps)
      本实现: (num_steps, num_envs) — 更利于 scan

    Args:
        rewards:     (T, N) 每步奖励
        dones:       (T, N) 终止标志（0 或 1）
        values:      (T, N) V(s_t)
        next_values: (T, N) V(s_{t+1})
        gamma:       折扣因子
        lmbda:       GAE λ

    Returns:
        advantages: (T, N) 优势估计
        returns:    (T, N) 回报 = advantages + values
    """
    not_done = 1.0 - dones

    # TD 残差
    deltas = rewards + gamma * next_values * not_done - values  # (T, N)

    def _scan_fn(gae, t_data):
        """反向扫描的单步。"""
        delta, nd = t_data
        gae = delta + gamma * lmbda * nd * gae
        return gae, gae

    # 反转时间维度，scan 从后往前
    deltas_reversed = jnp.flip(deltas, axis=0)      # (T, N)
    not_done_reversed = jnp.flip(not_done, axis=0)   # (T, N)

    init_gae = jnp.zeros_like(deltas[0])  # (N,)

    _, advantages_reversed = jax.lax.scan(
        _scan_fn,
        init_gae,
        (deltas_reversed, not_done_reversed),
    )

    # 反转回正常时间顺序
    advantages = jnp.flip(advantages_reversed, axis=0)  # (T, N)
    returns = advantages + values

    return advantages, returns
