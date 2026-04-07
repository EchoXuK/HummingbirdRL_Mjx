"""
utils/value_norm.py
值归一化（纯函数式 JAX 实现，支持 jit/vmap）

迁移自: training/scripts/utils.py → ValueNorm

原理：
  使用指数移动平均（EMA）跟踪返回值的均值和方差，
  用于归一化/反归一化 critic 输出。

  running_mean    ← β * running_mean + (1-β) * batch_mean
  running_mean_sq ← β * running_mean_sq + (1-β) * batch_sq_mean
  debiasing_term  ← β * debiasing_term + (1-β) * 1.0

  debiased_mean = running_mean / max(debiasing_term, ε)
  debiased_var  = max(debiased_mean_sq - debiased_mean², 0.01)

函数式 API：
  state = value_norm_init()
  state = value_norm_update(state, values)
  normalized = value_norm_normalize(state, values)
  denormalized = value_norm_denormalize(state, values)
"""

import jax.numpy as jnp
from typing import NamedTuple


class ValueNormState(NamedTuple):
    """值归一化的运行状态。"""
    running_mean: jnp.ndarray      # (1,)
    running_mean_sq: jnp.ndarray   # (1,)
    debiasing_term: jnp.ndarray    # ()


def value_norm_init() -> ValueNormState:
    """初始化值归一化状态。"""
    return ValueNormState(
        running_mean=jnp.zeros(1),
        running_mean_sq=jnp.zeros(1),
        debiasing_term=jnp.array(0.0),
    )


def _running_mean_var(
    state: ValueNormState,
    eps: float = 1e-5,
):
    """计算去偏后的均值和方差。"""
    debiased_mean = state.running_mean / jnp.maximum(state.debiasing_term, eps)
    debiased_mean_sq = state.running_mean_sq / jnp.maximum(state.debiasing_term, eps)
    debiased_var = jnp.maximum(debiased_mean_sq - debiased_mean ** 2, 1e-2)
    return debiased_mean, debiased_var


def value_norm_update(
    state: ValueNormState,
    input_vector: jnp.ndarray,
    beta: float = 0.995,
) -> ValueNormState:
    """
    用新一批数据更新运行统计量。

    对应原代码 ValueNorm.update()。

    Args:
        state:        当前 ValueNormState
        input_vector: (..., 1) 或 (...,) 输入值
        beta:         EMA 衰减系数

    Returns:
        更新后的 ValueNormState
    """
    # 展平除最后一维外的所有维度
    flat = input_vector.reshape(-1)
    batch_mean = jnp.mean(flat)
    batch_sq_mean = jnp.mean(flat ** 2)

    new_running_mean = state.running_mean * beta + batch_mean * (1.0 - beta)
    new_running_mean_sq = state.running_mean_sq * beta + batch_sq_mean * (1.0 - beta)
    new_debiasing_term = state.debiasing_term * beta + 1.0 * (1.0 - beta)

    return ValueNormState(
        running_mean=new_running_mean,
        running_mean_sq=new_running_mean_sq,
        debiasing_term=new_debiasing_term,
    )


def value_norm_normalize(
    state: ValueNormState,
    input_vector: jnp.ndarray,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """
    归一化输入值。

    对应原代码 ValueNorm.normalize()。

    Args:
        state:        当前 ValueNormState
        input_vector: (...) 输入值

    Returns:
        (...) 归一化后的值
    """
    mean, var = _running_mean_var(state, eps)
    return (input_vector - mean) / jnp.sqrt(var)


def value_norm_denormalize(
    state: ValueNormState,
    input_vector: jnp.ndarray,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """
    反归一化输入值。

    对应原代码 ValueNorm.denormalize()。

    Args:
        state:        当前 ValueNormState
        input_vector: (...) 归一化后的值

    Returns:
        (...) 原始尺度的值
    """
    mean, var = _running_mean_var(state, eps)
    return input_vector * jnp.sqrt(var) + mean
