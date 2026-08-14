"""EvoQuant formula baseline — 20-day cross-sectional reversal.

A DETERMINISTIC multi-factor score: no model is fitted, so there is no seed
randomness and no training cost. Evidence strength comes from stability across
subperiods rather than across seeds.

Contract: identical to the model candidates — reads --split-config, prints one
`FAMOU_RESULT {...}` line — so formula and model candidates compete in the
same pool under the same protocol.
"""

import argparse
import json
import math


HYPERPARAMS = {
    'terms': [
        {'feature': 'ROC20', 'weight': -1.0, 'transform': 'zscore'},
    ],
}


def load_split_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_universe(split_cfg, params):
    """Weighted combination of cross-sectionally normalised Alpha158 factors.

    The normalisation is applied by the shared runner so every formula uses the
    same convention: raw factor scales differ by orders of magnitude, and an
    un-normalised weighted sum would be decided by whichever column is largest
    rather than by the weights.
    """
    from famou_candidate_runtime import combine_factors  # provided by evaluator env

    return combine_factors(split_cfg, params["terms"])


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

    res = score_universe(cfg, HYPERPARAMS)
    ics = daily_rank_ic(res)
    if not ics:
        print('FAMOU_RESULT {"validity": 0.0, "error_info": "no valid IC days"}')
        return

    mean_ic = sum(ics) / len(ics)
    std_ic = math.sqrt(sum((v - mean_ic) ** 2 for v in ics) / max(1, len(ics) - 1))
    icir = (mean_ic / std_ic) if std_ic > 0 else 0.0
    out = {
        "validity": 1.0,
        "combined_score": mean_ic,
        "rank_ic": mean_ic,
        "rank_ic_std": std_ic,
        "icir": icir,
        # One element on purpose: the candidate is deterministic, so a second
        # "seed" would be the identical number. Stability lives in subperiods.
        "per_seed_rank_ic": [mean_ic],
        "per_seed_icir": [icir],
        "subperiod_rank_ic": res.get("subperiod_rank_ic", []),
        "deterministic": True,
        "n_ic_days": len(ics),
        "factors_used": res.get("factors_used", []),
    }
    print("FAMOU_RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
