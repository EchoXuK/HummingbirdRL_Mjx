#!/bin/bash

# ==========================================
# MJX NavRL 训练启动脚本 (带显存防碰撞保护)
# ==========================================

# --- 配置区 ---
TARGET_GPU=2            # 指定要使用的 GPU 编号
VRAM_THRESHOLD_MB=2048  # 安全阈值 (MiB)：如果占用超过 2GB，判定为有人在使用

echo "🔍 正在检查 GPU $TARGET_GPU 的显存占用情况..."

# 使用 nvidia-smi 提取指定 GPU 的已用显存 (忽略单位，只取纯数字)
USED_VRAM=$(nvidia-smi -i $TARGET_GPU --query-gpu=memory.used --format=csv,noheader,nounits)

# 容错检查：如果 nvidia-smi 挂了或者卡号不存在
if [ -z "$USED_VRAM" ]; then
    echo "❌ 错误: 无法获取 GPU $TARGET_GPU 的状态，请检查驱动或显卡编号。"
    exit 1
fi

echo "📊 GPU $TARGET_GPU 当前已用显存: ${USED_VRAM} MiB"

# --- 核心拦截逻辑 ---
if [ "$USED_VRAM" -gt "$VRAM_THRESHOLD_MB" ]; then
    echo "🚫 拦截警告: GPU $TARGET_GPU 当前占用 (${USED_VRAM} MiB) 超过了安全阈值 (${VRAM_THRESHOLD_MB} MiB)！"
    echo "🚫 为避免挤崩溃其他同学的程序，启动已自动中止。"
    echo "💡 建议: 请使用 nvitop 或 nvidia-smi 查看是谁在用，或者修改脚本中的 TARGET_GPU 换一张空闲卡。"
    exit 1
else
    echo "✅ 检查通过: GPU $TARGET_GPU 处于空闲状态。"
fi

# --- 训练启动区 ---
echo "⚙️  正在配置 JAX 环境变量..."

# 1. 开启 JAX 本地持久化编译缓存
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=10
export JAX_DEFAULT_MATMUL_PRECISION=highest
# 2. 极限压榨：强制 JAX 预分配 95% 的可用显存
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

echo "🚀 正在 GPU $TARGET_GPU 上启动后台训练任务 (4096 environments)..."

# 3. 指定 GPU 并使用 nohup 后台挂起运行
CUDA_VISIBLE_DEVICES=$TARGET_GPU nohup python training/train.py --route full --num_envs 4096 > train_full.log 2>&1 &

PID=$!
echo "✨ 训练已成功在后台启动！(进程 PID: $PID)"
echo "------------------------------------------------"
echo "👀 实时查看训练日志请运行："
echo "tail -f train_full.log"
echo "------------------------------------------------"