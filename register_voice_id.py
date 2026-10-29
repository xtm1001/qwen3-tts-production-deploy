# ==============================================================================
# Qwen3-TTS Voice ID Registration Script
# 
# 用于从参考音频中提取声纹特征 (Voice ID) 并保存为 .pkl 文件。
# 强制使用 CPU 和 float32 精度，确保提取过程的绝对稳定性。
#
# 使用方法:
#   python register_voice_id.py --model_path /data/models/... --ref_audio ... --ref_text "..." --voice_id xtm1001_voice
#
# License: MIT
# Author: 小天魔 (xtm1001)
# Repository: https://github.com/xtm1001/qwen3-tts-production-deploy
# ==============================================================================

import os
import argparse
import torch
from qwen_tts import Qwen3TTSModel

def main():
    parser = argparse.ArgumentParser(description="Register Voice ID from reference audio")
    parser.add_argument("--model_path", type=str, required=True, help="Base 模型路径 (如 /data/models/Qwen3-TTS-12Hz-1.7B-Base)")
    parser.add_argument("--ref_audio", type=str, required=True, help="参考音频路径")
    parser.add_argument("--ref_text", type=str, required=True, help="参考音频对应的文本内容")
    parser.add_argument("--voice_id", type=str, required=True, help="Voice ID 名称 (不含 .pkl)")
    parser.add_argument("--output_dir", type=str, default="./reference_audio", help="输出目录 (默认 ./reference_audio)")
    
    args = parser.parse_args()
    
    # 强制使用 CPU 和 float32，确保稳定性
    device = "cpu"
    dtype = "float32"
    
    print(f" 加载模型: {args.model_path}")
    print(f"   设备: {device} | 精度: {dtype}")
    
    model = Qwen3TTSModel.from_pretrained(
        args.model_path, 
        device_map=device, 
        dtype=dtype
    )
    
    print("✅ 模型加载完成，开始提取 Voice ID...")
    
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        x_vector_only_mode=False
    )
    
    if prompt_items is None:
        raise ValueError("模型返回的 prompt_items 为空")
    
    os.makedirs(args.output_dir, exist_ok=True)
    pkl_path = os.path.join(args.output_dir, f"{args.voice_id}.pkl")
    
    torch.save(prompt_items, pkl_path)
    
    if os.path.exists(pkl_path) and os.path.getsize(pkl_path) > 100:
        print(f"✅ Voice ID 注册成功！")
        print(f"   保存路径: {pkl_path}")
        print(f"   文件大小: {os.path.getsize(pkl_path)} bytes")
    else:
        raise IOError("文件保存失败或文件过小")

if __name__ == "__main__":
    main()
