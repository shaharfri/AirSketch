# Intel AI PC (NPU) Setup

## Prerequisites

- Intel Core Ultra processor (Meteor Lake or later)
- Windows 11 or Ubuntu 22.04+
- Intel NPU driver installed

## NPU Driver Installation

### Windows
1. Download the latest Intel NPU driver from:
   https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html
2. Run the installer and reboot.

### Linux (Ubuntu)
```bash
sudo apt install intel-driver-compiler-npu
```

## Setup

```bash
git clone https://github.com/Matan341/Skysketch.git && cd Skysketch
chmod +x setup_env.sh && ./setup_env.sh
micromamba activate skysketch
```

The setup script will automatically detect available devices and show NPU if present.

## Verify NPU is Available

```python
import openvino as ov
core = ov.Core()
print(core.available_devices)
# Expected: ['CPU', 'GPU', 'NPU']
```

## LLM Model Setup

Convert Qwen2.5-1.5B for OpenVINO:

```bash
pip install optimum[openvino]
optimum-cli export openvino \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --weight-format int4 \
    models/qwen2.5-1.5b-ov
```

## Running with NPU

```bash
# Both CNN and LLM on NPU
python -m skysketch.main --llm-device NPU --cnn-device NPU

# LLM on NPU, CNN on AUTO (auto-selects best)
python -m skysketch.main --llm-device NPU

# CPU only (default, always works)
python -m skysketch.main
```

## CLI Device Options

| Flag | Default | Options | What it controls |
|------|---------|---------|-----------------|
| `--llm-device` | CPU | CPU, GPU, NPU, AUTO | Qwen2.5-1.5B inference |
| `--cnn-device` | AUTO | CPU, GPU, NPU, AUTO | Sketch CNN classifier |
| `--no-llm` | false | — | Disable LLM entirely |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| NPU not in device list | Install/update NPU driver, reboot |
| LLM fails on NPU | Try `--llm-device GPU` or `--llm-device CPU` as fallback |
| "Model not found" | Run the optimum-cli export command above |
| Slow first inference | Normal — OpenVINO compiles the model on first run |

## Fallback

If NPU is unavailable, use `"CPU"` or `"AUTO"` (auto-selects best available).
The application works identically on CPU — only inference speed differs.
