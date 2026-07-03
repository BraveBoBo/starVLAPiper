#!/bin/bash
# aloha_piper (aloha-agilex dual-arm) 训练启动 — Qwen3.5-0.8B + QwenPI_v3, 16G 单卡可跑(zero2 + optimizer CPU offload)
# 用法(必须在 starVLA 目录跑, 即本仓的 smooth/starVLA):
#   bash examples/realRobots/Piper/train_files/run_aloha_piper_train.sh                       # 正式训练
#   bash examples/realRobots/Piper/train_files/run_aloha_piper_train.sh --trainer.max_train_steps=5   # 冒烟测
# 额外的 OmegaConf 覆盖直接追加(注意: dotlist 覆盖必须带 -- 前缀, 见 normalize_dotlist_args)
set -e
ENV=/home/oem/miniconda3/envs/starVLA   # ← 换机器改这里

# —— 跑通 fold 训练踩过的 6 个坑, 全在这几行环境变量里 ——
export PYTHONPATH="$PWD"                                       # 坑1: import starVLA 需要 cwd 进 path
export PATH="$ENV/bin:$PATH"                                   # 坑5: DeepSpeedCPUAdam 编译要 ninja(在 env/bin)
export CUDA_HOME="$ENV"
export LIBRARY_PATH="$ENV/targets/x86_64-linux/lib:$LIBRARY_PATH"        # 坑6: cpu_adam 链接 libcudart(conda 放 targets/lib)
export LD_LIBRARY_PATH="$ENV/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
export PYTHONNOUSERSITE=1                                      # 避免 ~/.local 干扰
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True        # 减显存碎片

# 坑3: 用 deepspeed config 启动(否则 dist.get_rank 时进程组未初始化)
# 坑4: fold_zero2_offload.yaml 把 AdamW 状态 offload 到 CPU, 解 16G 单卡 optimizer.step OOM
"$ENV/bin/accelerate" launch \
  --config_file starVLA/config/deepseeds/fold_zero2_offload.yaml \
  starVLA/training/train_starvla.py \
  --config_yaml examples/realRobots/Piper/train_files/starvla_train_aloha_piper.yaml \
  "$@"
