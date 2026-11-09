"""Metryki: czas podróży (średnia, p95), throughput, zatłoczenie, adaptacja, koszt obliczeniowy"""
import numpy as np


def adaptation_time(arr_t, tts, t_close, t_end, bin_w=30.0, tol=0.15, stable=3):
    """Czas od zamknięcia drogi do ustabilizowania się średniego czasu podróży.

    Poziom typowy = średnia z 4 przedziałów przed zamknięciem (+tol). Szukamy pierwszego
    przedziału po zamknięciu, w którym czas przekracza próg (skutek zamknięcia), a następnie
    pierwszych `stable` kolejnych przedziałów poniżej progu. 0 = brak wpływu, nan = brak powrotu.
    """
    arr_t, tts = np.asarray(arr_t), np.asarray(tts)
    nb = int(np.ceil(t_end / bin_w))
    idx = np.minimum((arr_t // bin_w).astype(int), nb - 1)
    means = np.full(nb, np.nan)
    for b in range(nb):
        sel = tts[idx == b]
        if len(sel):
            means[b] = sel.mean()
    kc = int(t_close // bin_w)
    base = means[max(0, kc - 4):kc]
    if not np.isfinite(base).any():
        return float("nan")
    thr = np.nanmean(base) * (1 + tol)
    post = means[kc:]
    impact = [i for i, x in enumerate(post) if np.isfinite(x) and x > thr]
    if not impact:
        return 0.0
    for j in range(impact[0] + 1, len(post) - stable + 1):
        w = post[j:j + stable]
        if np.all(np.isfinite(w)) and np.all(w <= thr):
            return float((kc + j) * bin_w - t_close)
    return float("nan")


def compute_metrics(sim) -> dict:
    cfg, T = sim.cfg, sim.t
    meas = [c for c in sim.cars if c.spawn_t >= cfg.warmup]
    done = [c for c in meas if c.arrive_t is not None]
    fin = [c for c in sim.cars if c.arrive_t is not None]
    tt = np.array([c.arrive_t - c.spawn_t for c in done])
    tt_cens = np.array([(c.arrive_t if c.arrive_t is not None else T) - c.spawn_t for c in meas])
    n_arr = sum(1 for c in fin if c.arrive_t >= cfg.warmup)
    minutes = max((T - cfg.warmup) / 60.0, 1e-9)
    n = max(sim.util_n, 1)
    m = {
        "mean_tt": float(tt.mean()) if len(tt) else float("nan"),
        "p95_tt": float(np.percentile(tt, 95)) if len(tt) else float("nan"),
        "mean_tt_censored": float(tt_cens.mean()) if len(tt_cens) else float("nan"),
        "throughput_per_min": n_arr / minutes,
        "completion_rate": len(done) / max(len(meas), 1),
        "unfinished": len(meas) - len(done),
        "mean_util": sim.util_sum / n,
        "max_util": sim.util_max,
        "overload_frac": sim.util_over / n,
        "router_time_us": 1e6 * sim.router_time / max(sim.n_decisions, 1),
        "storage_entries": sim.router.storage_entries(),
        "decision_mem_kb": sim.router.decision_memory_bytes() / 1024.0,
    }
    if cfg.closure_time is not None:
        m["adaptation_time"] = adaptation_time(
            [c.arrive_t for c in fin], [c.arrive_t - c.spawn_t for c in fin],
            cfg.closure_time, T)
    return m