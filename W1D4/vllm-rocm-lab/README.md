# vLLM ROCm Lab Notes

This folder adapts the vLLM lab for an AMD GPU machine.

## What changed from the original lab

- Use `python3` instead of `python` on this machine.
- Use `rocm-smi` instead of `nvidia-smi`.
- Use `vllm/vllm-openai-rocm:latest` instead of the default CUDA-oriented `pip install vllm` path.
- Use `Qwen/Qwen3-0.6B` by default because the detected discrete AMD GPU has about 8 GB VRAM.
- On this machine, Docker Desktop cannot pass `/dev/kfd` and `/dev/dri` into containers. Use native Docker Engine or host-native vLLM ROCm, not Ollama, for this lab.

## Run vLLM

### Host-native ROCm path used on this machine

The working setup is installed in `.venv-vllm` with `vllm-0.24.0+rocm723`.

```bash
source W1D4/vllm-rocm-lab/env_vllm_rocm.sh
.venv-vllm/bin/python -c "import torch, vllm; print(torch.version.hip); print(torch.cuda.get_device_name(0)); print(vllm.__version__)"
./W1D4/vllm-rocm-lab/run_vllm_rocm_host.sh
```

In another terminal:

```bash
curl http://127.0.0.1:8000/v1/models
```

### Docker ROCm path

The Docker path needs ROCm device files to be visible to Docker:

```bash
ls -l /dev/kfd /dev/dri
```

Start the OpenAI-compatible server:

```bash
cd W1D4/vllm-rocm-lab
./run_vllm_rocm.sh
curl http://127.0.0.1:8000/v1/models
```

Stop it:

```bash
./stop_vllm_rocm.sh
```

If the script says `/dev/kfd` or `/dev/dri` is missing, the current Docker context cannot access the AMD GPU. Use a native Linux Docker Engine with ROCm device passthrough rather than Docker Desktop.

## Run clients

```bash
.venv/bin/python W1D4/vllm-rocm-lab/client_chat.py
.venv/bin/python W1D4/vllm-rocm-lab/client_stream.py
.venv/bin/python W1D4/vllm-rocm-lab/rag_faiss.py
```

To override the vLLM endpoint or model:

```bash
LLM_BASE_URL=http://127.0.0.1:8000/v1 LLM_MODEL=Qwen/Qwen3-0.6B .venv/bin/python W1D4/vllm-rocm-lab/client_chat.py
```
