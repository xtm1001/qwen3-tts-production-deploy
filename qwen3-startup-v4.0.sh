#!/bin/bash
# ==============================================================================
# Qwen3-TTS/ASR Production Deploy - Startup Script
# 
# 生产级 Qwen3-TTS & ASR 私有化部署一键启动脚本。
# 支持多实例混合部署、GPU 智能轮转调度。
#
# License: MIT
# Author: 小天魔 (xtm1001)
# Repository: https://github.com/xtm1001/qwen3-tts-production-deploy
# ==============================================================================

set -uo pipefail

# ============ 配置区 ============
IMAGE_NAME="qwen3-tts:dual-mode-v2"
RESTART_POLICY="on-failure:3"

# 宿主机路径配置 (基于脚本所在目录的相对路径，方便 clone 后直接使用)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_BASE_DIR="${SCRIPT_DIR}/models"
APP_SCRIPT="${SCRIPT_DIR}/app_dual.py"
REFERENCE_AUDIO_DIR="${SCRIPT_DIR}/reference_audio"

# GPU 分配池 (根据实际 GPU 数量修改，此处为 8 卡示例)
GPU_POOL=("0" "1" "2" "3" "4" "5" "6" "7")

# ============ 模型类型映射 ============
declare -A MODEL_MAP
MODEL_MAP[custom]="custom_1.7b:Qwen3-TTS-12Hz-1.7B-CustomVoice:18880"
MODEL_MAP[design]="design_1.7b:Qwen3-TTS-12Hz-1.7B-VoiceDesign:18890"
MODEL_MAP[base]="base_1.7b:Qwen3-TTS-12Hz-1.7B-Base:18900"
MODEL_MAP[06b]="custom_0.6b:Qwen3-TTS-12Hz-0.6B-CustomVoice:18910"
MODEL_MAP[0.6b]="custom_0.6b:Qwen3-TTS-12Hz-0.6B-CustomVoice:18910"
MODEL_MAP[asr]="Qwen3-ASR-1.7B:Qwen3-ASR-1.7B:19000:qwenllm/qwen3-asr:latest"

# ============ 参数解析 ============
if [ $# -lt 2 ] || [ $(( $# % 2 )) -ne 0 ]; then
    echo "用法: $0 <模型类型> <数量> [<模型类型> <数量> ...]"
    echo ""
    echo "模型类型:"
    for key in "${!MODEL_MAP[@]}"; do
        IFS=':' read -r model_key dir _ _ <<< "${MODEL_MAP[$key]}"
        echo "  $key  →  $model_key"
    done
    echo ""
    echo "示例:"
    echo "  $0 custom 2 design 1 base 1 asr 1"
    echo "  $0 base 1"
    exit 1
fi

declare -A LAUNCH_PLAN
while [ $# -gt 0 ]; do
    MODEL_TYPE="$1"
    COUNT="$2"
    if [ -z "${MODEL_MAP[$MODEL_TYPE]+x}" ]; then
        echo "❌ 未知模型类型: $MODEL_TYPE"
        echo "   可选: ${!MODEL_MAP[*]}"
        exit 1
    fi
    if ! [[ "$COUNT" =~ ^[0-9]+$ ]] || [ "$COUNT" -eq 0 ]; then
        echo "❌ 数量必须是正整数，当前: $COUNT (模型: $MODEL_TYPE)"
        exit 1
    fi
    LAUNCH_PLAN[$MODEL_TYPE]=$COUNT
    shift 2
done

# ============ 前置检查 ============
echo "🔍 检查环境..."
[ ! -f "$APP_SCRIPT" ]    && { echo "❌ 找不到 $APP_SCRIPT"; exit 1; }
[ ! -d "$MODEL_BASE_DIR" ] && { echo "❌ 找不到模型目录 $MODEL_BASE_DIR"; exit 1; }

if [ ! -d "$REFERENCE_AUDIO_DIR" ]; then
    echo "📁 创建参考音频目录: $REFERENCE_AUDIO_DIR"
    mkdir -p "$REFERENCE_AUDIO_DIR"
fi

docker info >/dev/null 2>&1 || { echo "❌ Docker 未运行"; exit 1; }

echo ""
echo "📋 启动计划:"
TOTAL_COUNT=0
for MODEL_TYPE in "${!LAUNCH_PLAN[@]}"; do
    COUNT=${LAUNCH_PLAN[$MODEL_TYPE]}
    IFS=':' read -r MODEL_KEY _ BASE_PORT _ <<< "${MODEL_MAP[$MODEL_TYPE]}"
    echo "   $MODEL_TYPE × $COUNT (MODEL_KEY=$MODEL_KEY, 起始端口=$BASE_PORT)"
    TOTAL_COUNT=$((TOTAL_COUNT + COUNT))
done
echo "   总计: $TOTAL_COUNT 个容器"
echo ""

# ============ 清理旧容器 ============
echo "🧹 清理现有 tts/asr 容器..."
EXISTING=$(docker ps -a --format '{{.Names}}' | grep -E '^(tts-|asr-)' || true)
if [ -n "$EXISTING" ]; then
    echo "$EXISTING" | while read -r cname; do
        echo "   删除 $cname"
        docker rm -f "$cname" 2>/dev/null || true
    done
fi

# ============ 启动容器 ============
echo ""
echo " 开始启动容器..."
GPU_IDX=0
FAILED=0
SUCCESS=0

for MODEL_TYPE in "${!LAUNCH_PLAN[@]}"; do
    COUNT=${LAUNCH_PLAN[$MODEL_TYPE]}
    IFS=':' read -r MODEL_KEY EXPECTED_DIR BASE_PORT IMAGE_OVERRIDE <<< "${MODEL_MAP[$MODEL_TYPE]}"
    
    CURRENT_IMAGE="${IMAGE_OVERRIDE:-$IMAGE_NAME}"
    
    if [ ! -d "$MODEL_BASE_DIR/$EXPECTED_DIR" ]; then
        echo "️ 警告：模型目录不存在 $MODEL_BASE_DIR/$EXPECTED_DIR"
    fi
    
    for ((i=1; i<=COUNT; i++)); do
        if [ "$MODEL_TYPE" = "asr" ]; then
            NAME="asr-${i}"
        else
            NAME="tts-${MODEL_TYPE}-${i}"
        fi
        PORT=$((BASE_PORT + i - 1))
        GPU="${GPU_POOL[$GPU_IDX]}"
        
        echo "🚀 启动 $NAME (端口:$PORT MODEL_KEY:$MODEL_KEY GPU:$GPU)"
        
        if [ "$MODEL_TYPE" = "asr" ]; then
            if docker run -d \
                --name "$NAME" \
                --gpus "device=$GPU" \
                --restart="$RESTART_POLICY" \
                -p "$PORT:$PORT" \
                -v "$MODEL_BASE_DIR:/data/models" \
                -e "VLLM_ALIGNER_PATH=/data/models/Qwen3-ForcedAligner-0.6B" \
                -e "TRANSFORMERS_OFFLINE=1" \
                -e "HF_DATASETS_OFFLINE=1" \
                -e "HF_HUB_OFFLINE=1" \
                -e "MODELSCOPE_CACHE=/data/models" \
                -w /app \
                --shm-size="4g" \
                --health-cmd="curl -f http://localhost:$PORT/health || exit 1" \
                --health-interval=30s \
                --health-timeout=10s \
                --health-retries=3 \
                --health-start-period=120s \
                "$CURRENT_IMAGE" \
                bash -c "qwen-asr-serve /data/models/$EXPECTED_DIR \
                    --served-model-name Qwen3-ASR-1.7B \
                    --dtype float16 \
                    --enforce-eager \
                    --gpu-memory-utilization 0.90 \
                    --max-model-len 4096 \
                    --max-num-seqs 4 \
                    --host 0.0.0.0 \
                    --port $PORT"; then
                SUCCESS=$((SUCCESS + 1))
                echo "   ✅ $NAME 启动成功"
            else
                FAILED=$((FAILED + 1))
                echo "   ❌ $NAME 启动失败"
            fi
        else
            if docker run -d \
                --name "$NAME" \
                --gpus "device=$GPU" \
                --restart="$RESTART_POLICY" \
                -p "$PORT:$PORT" \
                -v "$MODEL_BASE_DIR:/data/models" \
                -v "$APP_SCRIPT:/app/app_dual.py":ro \
                -v "$REFERENCE_AUDIO_DIR:/app/reference_audio" \
                -w /app \
                -e MODEL_KEY="$MODEL_KEY" \
                -e DEVICE="cuda:0" \
                -e DTYPE="float16" \
                -e TTS_MAX_CHARS="400" \
                -e PYTHONUNBUFFERED=1 \
                -e MAX_INFERENCE_SEC="0" \
                -e VOICE_ID_DIR="/app/reference_audio" \
                --shm-size="2g" \
                --health-cmd="curl -f http://localhost:$PORT/health || exit 1" \
                --health-interval=30s \
                --health-timeout=10s \
                --health-retries=3 \
                --health-start-period=120s \
                "$CURRENT_IMAGE" \
                uvicorn app_dual:app --host 0.0.0.0 --port "$PORT" --workers 1; then
                SUCCESS=$((SUCCESS + 1))
                echo "   ✅ $NAME 启动成功"
            else
                FAILED=$((FAILED + 1))
                echo "   ❌ $NAME 启动失败"
            fi
        fi
        GPU_IDX=$(( (GPU_IDX + 1) % ${#GPU_POOL[@]} ))
    done
done

echo ""
echo "========================================="
echo "✅ 启动完成！成功: $SUCCESS | 失败: $FAILED | 总计: $TOTAL_COUNT"
echo "========================================="
echo ""
echo "📋 查看所有容器:  docker ps -a -f name='tts-|asr-'"
echo "📋 查看日志:      docker logs -f tts-base-1"
echo ""
echo "🌐 访问地址示例:"
for MODEL_TYPE in "${!LAUNCH_PLAN[@]}"; do
    IFS=':' read -r _ _ BASE_PORT _ <<< "${MODEL_MAP[$MODEL_TYPE]}"
    if [ "$MODEL_TYPE" = "asr" ]; then
        echo "   ASR:  http://localhost:$BASE_PORT/v1/audio/transcriptions"
    else
        echo "   TTS ($MODEL_TYPE): http://localhost:$BASE_PORT/health"
    fi
done
