
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=1
export ALL_PROXY=http://mt:mtstudio@10.8.17.48:8777


python run_famou.py \
  -c examples/mlebench/ventilator-pressure-prediction/config.yaml \
  -p examples/mlebench/ventilator-pressure-prediction/init.py \
  -e examples/mlebench/evaluator.py