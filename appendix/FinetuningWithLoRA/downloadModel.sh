#!/bin/bash

# 定义下载目录
DIRNAME="gpt2/355M"
mkdir -p ${DIRNAME}

# 基础地址（镜像站）
BASE="https://hf-mirror.com/openai-community/gpt2-medium/resolve/main"

# 批量下载所有文件到指定文件夹
wget -P ${DIRNAME} ${BASE}/config.json
wget -P ${DIRNAME} ${BASE}/pytorch_model.bin
wget -P ${DIRNAME} ${BASE}/vocab.json
wget -P ${DIRNAME} ${BASE}/merges.txt
wget -P ${DIRNAME} ${BASE}/tokenizer_config.json

echo "✅ GPT2-medium 模型下载完成！路径：$DIRNAME"