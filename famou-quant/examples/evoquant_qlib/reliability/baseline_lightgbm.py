"""Certified baseline: the official qlib Alpha158 + LightGBM configuration.

Hyperparameters are verbatim from ``workflow_config_lightgbm_Alpha158.yaml``,
so the incumbent an evolved candidate must beat is the published reference
rather than a weak strawman. It obeys the same candidate contract as generated
candidates (reads --split-config, prints one FAMOU_RESULT line), which is what
lets the sealed gate re-train it on sealed_promotion to compute a margin.

Note ``bagging_freq``: the official yaml omits it, which silently makes its
``subsample`` a no-op. It is set to 1 here so the documented value is the one
that actually applies — a deviation worth disclosing rather than inheriting a
bug for the sake of literalism.
"""

import argparse
import json
import math


HYPERPARAMS = {
    'objective': 'mse',
    'learning_rate': 0.2,
    'num_leaves': 210,
    'max_depth': 8,
    'colsample_bytree': 0.8879,
    'subsample': 0.8789,
    'lambda_l1': 205.6999,
    'lambda_l2': 580.9768,
    'min_data_in_leaf': 20,
    'bagging_freq': 1,
}


def load_split_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_and_predict(split_cfg, params, seed):
    from famou_candidate_runtime import run_candidate

    return run_candidate(split_cfg, params, seed, family="gbdt")


def daily_rank_ic(result):
    from scipy.stats import rankdata
    import numpy as np

    by_date = {}
    for d, p, y in zip(result["date"], result["pred"], result["label"]):
        by_date.setdefault(d, []).append((p, y))
    ics = []
    for d, pairs in sorted(by_date.items()):
        if len(pairs) < 3:
            continue
        p = np.array([a for a, _ in pairs])
        y = np.array([b for _, b in pairs])
        if p.std() == 0 or y.std() == 0:
            continue
        ics.append(float(np.corrcoef(rankdata(p), rankdata(y))[0, 1]))
    return ics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-config", required=True)
    cfg = load_split_config(ap.parse_args().split_config)

    all_ics = []
    per_seed_mean = []
    per_seed_icir = []
    for seed in cfg.get("seed_list", [11]):
        res = train_and_predict(cfg, HYPERPARAMS, seed)
        ics = daily_rank_ic(res)
        if ics:
            m = sum(ics) / len(ics)
            s = math.sqrt(sum((v - m) ** 2 for v in ics) / max(1, len(ics) - 1))
            per_seed_mean.append(m)
            per_seed_icir.append((m / s) if s > 0 else 0.0)
            all_ics.extend(ics)

    if not all_ics:
        print('FAMOU_RESULT {"validity": 0.0, "error_info": "no valid IC days"}')
        return

    mean_ic = sum(all_ics) / len(all_ics)
    std_ic = math.sqrt(sum((v - mean_ic) ** 2 for v in all_ics) / max(1, len(all_ics) - 1))
    out = {
        "validity": 1.0,
        "combined_score": mean_ic,
        "rank_ic": mean_ic,
        "rank_ic_std": std_ic,
        "icir": (mean_ic / std_ic) if std_ic > 0 else 0.0,
        "per_seed_rank_ic": per_seed_mean,
        "per_seed_icir": per_seed_icir,
        "n_ic_days": len(all_ics),
    }
    print("FAMOU_RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
