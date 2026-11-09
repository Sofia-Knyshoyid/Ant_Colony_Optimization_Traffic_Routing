"""Eksperymenty: natężenie ruchu, zamknięcie drogi, rozmiar miasta, parametry ACO,
tablice hierarchiczne vs płaskie. Każdy punkt powtarzany dla wielu ziaren (seed)"""
import argparse
import csv
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from .city import generate_city
from .config import ACOConfig, CityConfig, SimConfig
from .metrics import compute_metrics
from .routers import make_router
from .traffic import Simulation

ROUTERS_ALL = ["static_dijkstra", "static_astar", "dynamic_dijkstra", "dynamic_astar", "aco_hier"]
METRICS = ["mean_tt", "p95_tt", "mean_tt_censored", "throughput_per_min", "completion_rate",
           "mean_util", "max_util", "overload_frac", "router_time_us", "storage_entries",
           "decision_mem_kb", "adaptation_time"]


def run_task(task):
    city = replace(task["city"], seed=task["seed"])
    G = generate_city(city)
    sim = Simulation(G, task["sim"],
                     lambda net, rng: make_router(task["router"], net, rng, task["aco"]),
                     seed=task["seed"])
    sim.run()
    m = compute_metrics(sim)
    m.update(exp=task["exp"], group=task["group"], router=task["label"], seed=task["seed"])
    return m


def tasks_for(exp, group, router, seeds, city, sim, aco, label=None):
    return [dict(exp=exp, group=group, router=router, label=label or router, seed=s,
                 city=city, sim=sim, aco=aco) for s in seeds]


# definicje eksperymentów
def exp_traffic(a, seeds):
    t = []
    for lvl in ["low", "medium", "peak"]:
        sim = SimConfig(duration=a.duration, warmup=a.warmup, traffic=lvl)
        for r in ROUTERS_ALL:
            t += tasks_for("traffic", lvl, r, seeds, CityConfig(), sim, ACOConfig())
    return t


def exp_closure(a, seeds):
    sim = SimConfig(duration=a.duration, warmup=a.warmup, traffic=a.level,
                    closure_time=a.duration / 2)
    return [x for r in ROUTERS_ALL
            for x in tasks_for("closure", a.level, r, seeds, CityConfig(), sim, ACOConfig())]


def exp_city_size(a, seeds):
    t = []
    for nx_, ny_ in [(2, 2), (3, 3), (4, 4)]:
        city = CityConfig(n_regions_x=nx_, n_regions_y=ny_)
        sim = SimConfig(duration=a.duration, warmup=a.warmup, traffic=a.level,
                        rate_scale=nx_ * ny_ / 9.0)
        for r in ROUTERS_ALL:
            t += tasks_for("city_size", f"{nx_}x{ny_}", r, seeds, city, sim, ACOConfig())
    return t


def exp_params(a, seeds):
    sim = SimConfig(duration=a.duration, warmup=a.warmup, traffic=a.level)
    t = tasks_for("params", "ref", "dynamic_astar", seeds, CityConfig(), sim, ACOConfig(),
                  label="dynamic_astar")
    sweeps = {"alpha": [0.5, 1.0, 2.0], "beta": [1.0, 2.0, 4.0],
              "rho": [0.01, 0.05, 0.2], "n_scouts": [0, 5, 20]}
    for name, values in sweeps.items():
        for v in values:
            aco = replace(ACOConfig(), **{name: v})
            t += tasks_for("params", f"{name}={v}", "aco_hier", seeds, CityConfig(), sim, aco,
                           label="aco_hier")
    return t


def exp_hier_flat(a, seeds):
    t = []
    for nx_, ny_ in [(2, 2), (3, 3), (4, 4)]:
        city = CityConfig(n_regions_x=nx_, n_regions_y=ny_)
        sim = SimConfig(duration=a.duration, warmup=a.warmup, traffic=a.level,
                        rate_scale=nx_ * ny_ / 9.0)
        for r in ["aco_hier", "aco_flat"]:
            t += tasks_for("hier_flat", f"{nx_}x{ny_}", r, seeds, city, sim, ACOConfig())
    return t


EXPERIMENTS = {"traffic": exp_traffic, "closure": exp_closure, "city_size": exp_city_size,
               "params": exp_params, "hier_flat": exp_hier_flat}


# agregacja i wykresy
def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["group"], r["router"])].append(r)
    out = []
    for (g, rt), rs in groups.items():
        row = {"group": g, "router": rt, "n_seeds": len(rs)}
        for m in METRICS:
            v = np.array([r.get(m, np.nan) for r in rs], float)
            ok = np.isfinite(v)
            row[m + "_mean"] = float(v[ok].mean()) if ok.any() else float("nan")
            row[m + "_std"] = float(v[ok].std()) if ok.any() else float("nan")
        out.append(row)
    return out


def save_csv(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def bar_plot(agg, metric, path, title=None):
    vals = [r[metric + "_mean"] for r in agg]
    if not np.isfinite(vals).any():
        return
    groups = list(dict.fromkeys(r["group"] for r in agg))
    routers = list(dict.fromkeys(r["router"] for r in agg))
    width = 0.8 / len(routers)
    fig, ax = plt.subplots(figsize=(1.5 + 1.6 * len(groups), 4))
    for k, rt in enumerate(routers):
        xs, ys, es = [], [], []
        for gi, g in enumerate(groups):
            for r in agg:
                if r["group"] == g and r["router"] == rt:
                    xs.append(gi + k * width)
                    ys.append(r[metric + "_mean"])
                    es.append(r[metric + "_std"])
        ax.bar(xs, ys, width, yerr=es, capsize=2, label=rt)
    ax.set_xticks(np.arange(len(groups)) + 0.4 - width / 2)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or metric)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exp", choices=list(EXPERIMENTS))
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--duration", type=float, default=1800.0)
    p.add_argument("--warmup", type=float, default=300.0)
    p.add_argument("--level", default="medium", choices=["low", "medium", "peak"])
    p.add_argument("--out", default="results")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = EXPERIMENTS[a.exp](a, list(range(a.seeds)))
    print(f"{len(tasks)} uruchomień...")
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            rows = list(ex.map(run_task, tasks))
    else:
        rows = [run_task(t) for t in tasks]
    agg = aggregate(rows)
    save_csv(rows, os.path.join(a.out, f"{a.exp}_raw.csv"))
    save_csv(agg, os.path.join(a.out, f"{a.exp}_agg.csv"))
    for m in METRICS:
        bar_plot(agg, m, os.path.join(a.out, f"{a.exp}_{m}.png"), f"{a.exp}: {m} (średnia ± std)")
    print("Zapisano wyniki w", a.out)


if __name__ == "__main__":
    main()