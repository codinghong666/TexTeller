# #!/usr/bin/env bash
# set -exu
# #!/bin/bash
# source /Users/coding_hong/Documents/anaconda/miniconda3/envs/html/bin/activate


# export CHECKPOINT_DIR="default"
# export TOKENIZER_DIR="default"

# streamlit run web.py


#!/usr/bin/env bash
set -exu

# 确保 Conda 可用
source /Users/coding_hong/Documents/anaconda/miniconda3/bin/activate

# 激活 `html` 环境
conda activate html

# 设置环境变量
export CHECKPOINT_DIR="default"
export TOKENIZER_DIR="default"

# 运行 Streamlit 应用
python -m streamlit run web.py
