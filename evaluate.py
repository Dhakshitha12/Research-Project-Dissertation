"""
Paired evaluation, lambda sweep, latency-cost frontier and statistical tests.

Protocol (proposal Sections 3.2 and 3.5):
  * paired design: every policy faces identical arrival arrays;
  * evaluation uses HELD-OUT generator families and test seeds disjoint
    from the training seeds (10_000+ vs 0..4);
  * Shapiro-Wilk on paired differences -> paired t-test (normal) or
    Wilcoxon signed-rank (otherwise);
  * Bonferroni correction over the three baseline comparisons per metric;
  * effect sizes: Cohen's d (parametric) or rank-biserial r (non-parametric);
  * regime-shift run: sparse -> burst -> sparse (slide requirement).
"""
from __future__ import annotations
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as st

from envs.workloads import (poisson_stream, bursty_stream, periodic_stream,
                            diurnal_stream, sparse_stream, azure_like_stream,
                            regime_shift_stream)
from envs.baselines import NoPreWarm, StaticWarm, ThresholdPolicy
from envs.workloads import load_function_profile

TEST_EPISODES = 2
EP_SECONDS = 3600
EVAL_LAM = 0.02
TEST_SEED_BASE = 10_000
N_BASELINES = 3


def heldout_arrivals():
    T = EP_SECONDS
    fams = {
        "poisson_ho":    lambda r: poisson_stream(T, r, rate=0.3),
        "bursty_ho":     lambda r: bursty_stream(T, r, burst_rate=7.0, p_burst=0.02),
        "periodic_ho":   lambda r: periodic_stream(T, r, period_s=420, peak=5.0),
        "diurnal_ho":    lambda r: diurnal_stream(T, r, peak=3.5),
        "sparse_ho":     lambda r: sparse_stream(T, r, rate=0.004),
        "azure_like_ho": lambda r: azure_like_stream(T, r),
    }
    out = {}
    for name, fn in fams.items():
        for ep in range(TEST_EPISODES):
            out[(name, ep)] = fn(np.random.default_rng(TEST_SEED_BASE + ep))
    return out


def load_policies(models_dir):
    import re
    import stable_baselines3 as sb3
    pols = []
    import shutil
    candidates = [q for q in sorted(glob.glob(os.path.join(models_dir, "*"))) if os.path.isfile(q)]
    for p in candidates:
        if not p.endswith(".zip"):
            z = p + ".zip"
            if not os.path.exists(z):
                shutil.copy(p, z)
            p = z
        base = os.path.basename(p)[:-4]
        m = re.match(r"(ppo|dqn)_lam([0-9.]+)_seed(\d+)", base)
        if not m:
            continue
        algo, lam = m.group(1), float(m.group(2))
        model = sb3.PPO.load(p) if algo == "ppo" else sb3.DQN.load(p)
        pols.append({"label": base, "kind": "rl", "algo": algo, "obj": model,
                     "lam": lam, "seed": int(m.group(3))})
    for b in [NoPreWarm(), StaticWarm(K=1), ThresholdPolicy()]:
        pols.append({"label": b.name, "kind": "baseline", "algo": b.name, "obj": b, "lam": EVAL_LAM})
    return pols


def evaluate_all(models_dir, out_dir, profile):
    from envs.serverless_env import run_policy_on_arrivals

    rows = []
    arrivals = heldout_arrivals()
    for pol in load_policies(models_dir):
        for (fam, ep), arr in arrivals.items():
            m = run_policy_on_arrivals(pol["obj"], arr, profile=profile, lam=pol["lam"])
            rows.append({"label": pol["label"], "kind": pol["kind"], "algo": pol["algo"],
                         "lam": pol["lam"], "seed": pol.get("seed"),
                         "family": fam, "episode": ep, **m})
        print(f"evaluated {pol['label']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "eval_rows.csv"), index=False)
    return df


# ------------------------------------------------------------------ stats
def paired_analysis(rl_vals, base_vals, comparisons=N_BASELINES, alpha=0.05):
    rl, ba = np.asarray(rl_vals, float), np.asarray(base_vals, float)
    d = rl - ba
    out = {"n_pairs": int(len(d)),
           "rl_mean": float(rl.mean()), "base_mean": float(ba.mean()),
           "mean_diff": float(d.mean())}
    if len(d) < 5 or np.allclose(d, 0):
        out.update({"test": "insufficient variation", "p_raw": None, "p_bonferroni": None})
        return out
    try:
        sw_p = float(st.shapiro(d).pvalue)
    except Exception:
        sw_p = 0.0
    if sw_p > alpha and d.std(ddof=1) > 0:
        p = float(st.ttest_rel(rl, ba).pvalue)
        out.update({"test": "paired t-test (Shapiro-Wilk p=%.3g" % sw_p,
                    "p_raw": p, "cohen_d": float(d.mean() / (d.std(ddof=1) + 1e-12))})
    else:
        p = float(st.wilcoxon(rl, ba).pvalue)
        ranks = st.rankdata(np.abs(d))
        r_rb = float(np.sum(ranks[d > 0]) / np.sum(ranks))
        out.update({"test": f"wilcoxon signed-rank (Shapiro-Wilk p={sw_p:.3g})",
                    "p_raw": p, "rank_biserial": r_rb})
    out["p_bonferroni"] = float(min(p * comparisons, 1.0))
    return out


def statistics(df, out_dir, primary_lam=0.02):
    sub = df[(np.isclose(df["lam"], primary_lam)) | (df["kind"] == "baseline")]
    # RL seed-mean per (family, episode); baselines are deterministic per (family, episode)
    rl = sub[sub["kind"] == "rl"].groupby(["algo", "family", "episode"])[
        ["cold_rate", "cost_usd", "p99_s", "instance_seconds"]].mean().reset_index()
    ba = sub[sub["kind"] == "baseline"].set_index(["algo", "family", "episode"])[
        ["cold_rate", "cost_usd", "p99_s", "instance_seconds"]]

    stats_rows = []
    for algo in sorted(rl["algo"].unique()):
        a = rl[rl["algo"] == algo].set_index(["family", "episode"])
        for baseline in sorted(ba.index.get_level_values(0).unique()):
            b = ba.xs(baseline, level="algo")
            idx = a.index.intersection(b.index)
            for metric in ["cold_rate", "cost_usd", "p99_s"]:
                res = paired_analysis(a.loc[idx, metric].values, b.loc[idx, metric].values)
                stats_rows.append({"rl_algo": algo, "baseline": baseline, "metric": metric, **res})
    sdf = pd.DataFrame(stats_rows)
    sdf.to_csv(os.path.join(out_dir, "stats.csv"), index=False)
    return sdf


# ---------------------------------------------------------------- frontier
def frontier(df, out_dir):
    metrics = ["cold_rate", "cost_usd", "p99_s", "instance_seconds", "gb_seconds"]
    grouped = df.groupby(["kind", "algo", "lam"], sort=True)
    points = []
    rng = np.random.default_rng(42)
    for (kind, algo, lam), group in grouped:
        if kind == "rl" and group["seed"].notna().any():
            case_keys = ["family", "episode"]
            cases = group[case_keys].drop_duplicates().sort_values(case_keys)
            seed_ids = sorted(group["seed"].dropna().unique())
            samples = np.stack([
                group[group["seed"] == seed].set_index(case_keys).loc[cases.set_index(case_keys).index,
                                                                      metrics].to_numpy()
                for seed in seed_ids
            ])
            means = samples.mean(axis=(0, 1))
            boot = np.empty((2000, len(metrics)))
            for i in range(len(boot)):
                seed_indices = rng.integers(0, len(seed_ids), size=len(seed_ids))
                case_indices = rng.integers(0, len(cases), size=len(cases))
                boot[i] = samples[seed_indices][:, case_indices, :].mean(axis=(0, 1))
        else:
            samples = group[metrics].to_numpy()
            means = samples.mean(axis=0)
            boot = np.empty((2000, len(metrics)))
            for i in range(len(boot)):
                picked = rng.integers(0, len(samples), size=len(samples))
                boot[i] = samples[picked].mean(axis=0)
        low, high = np.percentile(boot, [2.5, 97.5], axis=0)
        row = {
            "kind": kind,
            "algo": algo,
            "lam": lam,
            "n_seeds": group["seed"].nunique() if kind == "rl" else 0,
            "n_cases": len(cases) if kind == "rl" and group["seed"].notna().any()
                       else len(samples),
            "bootstrap_resamples": len(boot),
            "confidence_level": 0.95,
        }
        for j, metric in enumerate(metrics):
            row[metric] = float(means[j])
            row[f"{metric}_ci_low"] = float(low[j])
            row[f"{metric}_ci_high"] = float(high[j])
        points.append(row)
    agg = pd.DataFrame(points)
    agg.to_csv(os.path.join(out_dir, "frontier_points.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ppo = agg[agg["algo"] == "ppo"].sort_values("lam")
    if len(ppo):
        ax.plot(ppo["cost_usd"], ppo["cold_rate"], "o-", label="PPO (λ sweep)")
        ax.errorbar(
            ppo["cost_usd"], ppo["cold_rate"],
            xerr=np.maximum(0, np.vstack([
                ppo["cost_usd"] - ppo["cost_usd_ci_low"],
                ppo["cost_usd_ci_high"] - ppo["cost_usd"],
            ])),
            yerr=np.maximum(0, np.vstack([
                ppo["cold_rate"] - ppo["cold_rate_ci_low"],
                ppo["cold_rate_ci_high"] - ppo["cold_rate"],
            ])),
            fmt="none", ecolor="tab:blue", capsize=3
        )
        for _, r in ppo.iterrows():
            ax.annotate(f"λ={r['lam']:g}", (r["cost_usd"], r["cold_rate"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)
    for algo, style in [("dqn", "s--"), ("no_prewarm", "^"), ("static_warm", "v"), ("threshold", "D")]:
        g = agg[agg["algo"] == algo]
        if len(g):
            ax.errorbar(
                g["cost_usd"], g["cold_rate"],
                xerr=np.maximum(0, np.vstack([
                    g["cost_usd"] - g["cost_usd_ci_low"],
                    g["cost_usd_ci_high"] - g["cost_usd"],
                ])),
                yerr=np.maximum(0, np.vstack([
                    g["cold_rate"] - g["cold_rate_ci_low"],
                    g["cold_rate_ci_high"] - g["cold_rate"],
                ])),
                fmt=style, label=algo, capsize=3
            )
    ax.set_xlabel("cost (USD per episode-hour, GB-seconds × $/GB-s)")
    ax.set_ylabel("cold-start rate")
    ax.set_title("Latency–cost frontier on held-out functions (paired arrivals)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "frontier.png")
    fig.savefig(path, dpi=150)
    return agg


# ------------------------------------------------------- regime shift run
def regime_shift(df_models_dir, out_dir, profile):
    from envs.serverless_env import run_policy_on_arrivals

    stream = regime_shift_stream(T=3600, seed=7)
    segs = [(0, 1200, "sparse"), (1200, 2400, "burst"), (2400, 3600, "sparse")]
    rows, pols = [], load_policies(df_models_dir)
    for pol in pols:
        for s, e, name in segs:
            m = run_policy_on_arrivals(pol["obj"], stream[s:e], profile=profile, lam=pol["lam"])
            rows.append({"policy": pol["label"], "segment": name,
                         "cold_rate": m["cold_rate"], "cost_usd": m["cost_usd"],
                         "cold_count": m["cold_count"]})
    rdf = pd.DataFrame(rows)
    rdf.to_csv(os.path.join(out_dir, "regime_shift.csv"), index=False)
    return rdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="../models")
    ap.add_argument("--out", default="../results")
    ap.add_argument("--calibration", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "aws_calibration.csv"))
    ap.add_argument("--runtime", default="python3.12")
    ap.add_argument("--workload", default="cpu_loop")
    ap.add_argument("--memory-grid-mb", type=int, nargs="+")
    ap.add_argument("--require-full-protocol", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    profile = load_function_profile(args.calibration, args.runtime, args.workload,
                                    args.memory_grid_mb)
    if args.require_full_protocol:
        import re
        trained = [os.path.basename(p) for p in glob.glob(os.path.join(args.models, "*.zip"))]
        for algo in ("ppo", "dqn"):
            for lam in ([0.005, 0.01, 0.02, 0.05, 0.1] if algo == "ppo" else [0.02]):
                seeds = {
                    int(match.group(1))
                    for name in trained
                    if (match := re.fullmatch(
                        rf"{algo}_lam{re.escape(str(lam))}_seed(\d+)\.zip", name
                    ))
                }
                if seeds != {0, 1, 2, 3, 4}:
                    raise ValueError(
                        f"Expected {algo} λ={lam} seeds 0..4, found {sorted(seeds)}"
                    )

    df = evaluate_all(args.models, args.out, profile)
    sdf = statistics(df, args.out)
    frontier(df, args.out)
    rdf = regime_shift(args.models, args.out, profile)

    summary = {
        "episodes_per_family": TEST_EPISODES,
        "families": sorted(df["family"].unique().tolist()),
        "policies": sorted(df["label"].unique().tolist()),
        "headline_cold_rate_by_policy": df.groupby("label")["cold_rate"].mean().round(4).to_dict(),
        "headline_cost_usd_by_policy": df.groupby("label")["cost_usd"].mean().round(6).to_dict(),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== HEADLINE (held-out functions, mean over families/episodes) ===")
    print(df.groupby(["kind", "algo"])[["cold_rate", "p99_s", "instance_seconds", "cost_usd"]]
          .mean().round(4).to_string())
    print("\n=== STATISTICAL TESTS (lam=%.2g, Bonferroni x%d) ===" % (EVAL_LAM, N_BASELINES))
    print(sdf[sdf["metric"] == "cold_rate"].to_string(index=False))
    print("\n=== REGIME SHIFT (sparse -> burst -> sparse) ===")
    print(rdf.pivot_table(index="policy", columns="segment", values="cold_rate").round(4).to_string())


if __name__ == "__main__":
    main()
