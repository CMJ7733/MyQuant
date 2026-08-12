
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=1
export ALL_PROXY=http://mt:mtstudio@10.8.17.48:8777


python run_famou.py \
  -c examples/mlebench/siim-covid19-detection/config.yaml \
  -p examples/mlebench/siim-covid19-detection/init.py \
  -e examples/mlebench/evaluator.py