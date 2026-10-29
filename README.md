# Qwen3-TTS/ASR Production Deploy

[![GitHub](https://img.shields.io/badge/GitHub-xtm1001-181717?logo=github)](https://github.com/xtm1001/qwen3-tts-production-deploy)
[![Gitee](https://img.shields.io/badge/Gitee-xtm1001-C71D23?logo=gitee)](https://gitee.com/xtm1001/qwen3-tts-production-deploy)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

🚀 **Production-Grade Qwen3-TTS & ASR Private Deployment Solution**. Provides one-click multi-instance container management, intelligent GPU scheduling, and industrial-grade FastAPI interfaces for Text-to-Speech (TTS) and Automatic Speech Recognition (ASR).

This project bridges the gap between official basic scripts and enterprise production environments. It pioneers the **Voice ID Persistence Mechanism**, completely solving the tone fragmentation in long-text generation and timbre drift across requests. It also integrates ASR services to provide infrastructure support for building closed-loop voice interaction systems.

---

## ✨ Core Features

*   **Voice ID Persistence (Core)**: Supports pre-registering voiceprint features (`.pkl`). Register once, reuse forever, achieving **absolute timbre consistency across days and requests**, while significantly accelerating long-text inference.
*   **One-Click Hybrid Deployment**: Supports launching multiple TTS (Custom/Design/Base) and ASR instances simultaneously via a single Shell command.
*   **Intelligent GPU Scheduling**: Built-in multi-card global round-robin allocation mechanism, automatically distributing instances evenly across different GPUs to maximize hardware utilization.
*   **Seamless Long-Text Splicing**: Built-in punctuation-based intelligent chunking algorithm and audio smooth transition, completely eliminating tone fragmentation in long-text generation.
*   **Industrial-Grade Robustness**: Non-blocking thread locks (to prevent VRAM overflow), client disconnection detection, inference timeout control, and detailed statistical monitoring.
*   **Advanced Sampling Control**: Exposes parameters like `seed`, `temperature`, and `repetition_penalty` for fine-grained control over generation effects, preventing model looping or stuttering.
*   **TTS-ASR Closed-Loop Support**: Integrates ASR services for automated verification of TTS generation quality (WER evaluation), and provides foundational support for building **TTS -> ASR -> LLM (Qwen)** end-to-end voice interaction loops.

---

## 📂 Directory Structure (External Mount Architecture)

This project adopts an industrial-grade architecture where **"Environment, Code, and Data are separated"**. The repository **does not contain** massive model weights; all data is mounted via Docker Volumes.

```text
qwen3-tts-production-deploy/
├── .gitignore                  # Prevents accidental upload of large files (ignores models/ directory)
├── LICENSE                     # MIT License
├── README.md                   # Project documentation (Chinese)
├── README.en.md                # Project documentation (English)
├── Dockerfile                  # Pure environment image build script
── app_dual.py                 # Core TTS API service script (externally mounted)
├── qwen3-startup-v4.0.sh       # One-click startup script
├── register_voice_id.py        # External Voice ID offline registration script (forced CPU execution for stability)
└── reference_audio/            # Shared audio and Voice ID directory (externally mounted)
    ├── example_ref.wav         # Example reference audio
    └── example_ref.pkl         # Example Voice ID file
```

---

## ⚡ Quick Start

### 1. Clone the Repository (Choose either platform)

**GitHub:**
```bash
git clone https://github.com/xtm1001/qwen3-tts-production-deploy.git
cd qwen3-tts-production-deploy
```
**Gitee (Recommended for users in China, faster speed):**
```bash
git clone https://gitee.com/xtm1001/qwen3-tts-production-deploy.git
cd qwen3-tts-production-deploy
```

### 2. Prepare Model Weights
This repository **does not contain** model weights. Please download the Qwen3-TTS models yourself and place them in the `models/` directory:
```bash
mkdir -p models
# Recommended to use ModelScope (faster in China)
pip install modelscope
modelscope download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir models/Qwen3-TTS-12Hz-1.7B-Base
# (Download CustomVoice, VoiceDesign, etc., as needed)
```

### 3. Build the Docker Image
```bash
# Build the pure environment image (approx. 5-8GB, does not contain model weights)
docker build -t qwen3-tts:dual-mode-v2 .
```
### 4. One-Click Service Startup
Grant execution permissions to the script and start the service combinations you need:
```bash
chmod +x qwen3-startup-v4.0.sh

# Example: Start 1 Base (clone), 1 Custom (preset), and 1 ASR instance
./qwen3-startup-v4.0.sh base 1 custom 1 asr 1
```

**Default Port Mapping:**
*   `18880`: Custom Mode (TTS)
*   `18890`: Design Mode (TTS)
*   `18900`: Base Mode (TTS Voice Cloning)
*   `19000`: ASR Mode (Speech Recognition)

---

## 📡 API Documentation

### 1. Text-to-Speech (Core)
**POST** `/v1/audio/speech`

#### Request Parameters (JSON Body)

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `text` | string | **Yes** | The text content to be synthesized. |
| `language` | string | No | Text language, default `"Chinese"`. |
| **Base Mode Specific** | | | |
| `ref_audio` | string | Conditional | Reference audio path (**absolute path inside container**, e.g., `/app/reference_audio/xxx.wav`). Required if `voice_id` is not provided. |
| `ref_text` | string | No | Text corresponding to the reference audio (highly recommended to improve ICL cloning quality). |
| `voice_id` | string | Conditional | Pre-registered Voice ID name (without `.pkl`). **Higher priority than `ref_audio`**. |
| **Advanced Control Parameters** | | | |
| `seed` | int | No | Random seed. Fixing this value ensures completely identical audio generation for the same input. The response header will return `X-Used-Seed`. |
| `temperature` | float | No | Sampling temperature (0.0-2.0). Lower is more stable, higher is more emotional. |
| `repetition_penalty`| float | No | Repetition penalty coefficient (default 1.05). **>1.0 effectively prevents model stuttering or looping**. |

#### Response
*   **Success**: Returns an `audio/wav` format audio stream.

### 2. Voice ID Registration (Base Mode Specific)
To ensure the absolute stability of voiceprint extraction, we provide an independent registration script `register_voice_id.py`. This script **forces execution on CPU with float32 precision**, completely avoiding precision overflow or operator compatibility issues that may occur in GPU environments.

**Usage:**
```bash
python register_voice_id.py \
    --model_path /data/models/Qwen3-TTS-12Hz-1.7B-Base \
    --ref_audio /app/reference_audio/my_voice.wav \
    --ref_text "Hello, I am the tester, glad to serve you." \
    --voice_id xtm1001_voice \
    --output_dir /app/reference_audio
```

After registration, when calling the `/v1/audio/speech` interface, you only need to pass the `voice_id` parameter to achieve second-level response and absolute consistency:
```json
{
  "text": "This is a test using the pre-stored Voice ID.",
  "voice_id": "xtm1001_voice",
  "seed": 42
}
```

### 3. Management and Monitoring
*   **GET** `/health`: Returns service status and model information.
*   **GET** `/ready`: Used for load balancer liveness probes.
*   **GET** `/stats`: Returns inference statistics (request count, average latency, last error, etc.).

---

## 🎙️ About the ASR Service

The ASR (Automatic Speech Recognition) service integrated into this project aims to build a complete voice processing pipeline.

*   **Image Source**: The ASR container uses the pre-built community/official image `qwenllm/qwen3-asr:latest`, which is **not built by this project itself**.
*   **Core Use Cases**:
    1.  **Automated Verification**: Used to back-test audio generated by TTS, calculating Word Error Rate (WER) to evaluate TTS model synthesis quality.
    2.  **Closed-Loop Interaction**: Provides infrastructure support for building **TTS (Speech Synthesis) -> ASR (Speech Recognition) -> Qwen (Large Language Model)** end-to-end voice interaction loops in the future.

---

## 💡 Best Practices

1.  **Long-Text Punctuation Standards**:
    *   The model has better support for Chinese punctuation. It is recommended to use Chinese ellipsis `……` and Chinese em dash `——` to control pauses.
    *   Avoid mixing English hyphens `-` or three dots `...` in Chinese text, as the model might misread them as characters.
2.  **Solving "Looping/Stuttering"**:
    *   If you find the model repeating a sentence when generating long text, please increase the `repetition_penalty` in the request (e.g., set to `1.1` or `1.2`).
3.  **Path Attention**:
    *   The `ref_audio` passed in the API request must be the **path inside the container** (e.g., `/app/reference_audio/xxx.wav`), not the host machine path (e.g., `/mnt/Block/...`).

---

## 🔄 Dual-Platform Maintenance

This project is maintained synchronously on both **GitHub** and **Gitee** to ensure the best access and cloning experience for developers worldwide.
*   **Primary Repository**: Gitee (Optimized for domestic network environments in China, smoother for downloading models and dependencies)
*   **Mirror Repository**: GitHub (Oriented towards the international community, facilitating Issue discussions and Star accumulation)

Developers can achieve one-click synchronous pushing by configuring dual remote Git repositories:
```bash
git remote add github https://github.com/xtm1001/qwen3-tts-production-deploy.git
git remote add gitee https://gitee.com/xtm1001/qwen3-tts-production-deploy.git
# Push by executing git push github main and git push gitee main respectively
```

---

## 🤝 Acknowledgements

*   This project is built upon the [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) open-source model.
*   The ASR service is based on the `qwenllm/qwen3-asr:latest` image.
*   AI-assisted programming tools (such as Tencent Yuanbao, Qwen, etc.) were used during the core architecture design and code implementation to significantly improve development efficiency. The core business logic and engineering transformations were independently completed by the author.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

**Author**: 小天魔 (xtm1001)  
**Repositories**: 
- GitHub: https://github.com/xtm1001/qwen3-tts-production-deploy
- Gitee: https://gitee.com/xtm1001/qwen3-tts-production-deploy
