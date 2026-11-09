"""Szybkie uruchomienie pojedynczej symulacji + opcjonalna mapa obciążenia"""
import argparse

import matplotlib.pyplot as plt

from aco_traffic.city import generate_city
from aco_traffic.config import ACOConfig, CityConfig, SimConfig
from aco_traffic.metrics import compute_metrics
from aco_traffic.routers import ROUTER_NAMES, make_router
from aco_traffic.traffic import Simulation
from aco_traffic.visualization import draw_city

p = argparse.ArgumentParser()
p.add_argument("--router", default="aco_hier", choices=ROUTER_NAMES)
p.add_argument("--traffic", default="medium", choices=["low", "medium", "peak"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--duration", type=float, default=1800)
p.add_argument("--closure", action="store_true", help="zamknij główną drogę w połowie symulacji")
p.add_argument("--scouts", type=int, default=0)
p.add_argument("--show", action="store_true")
p.add_argument("--animate", action="store_true")
a = p.parse_args()

city = CityConfig(seed=a.seed)
sim_cfg = SimConfig(duration=a.duration, traffic=a.traffic,
                    closure_time=a.duration / 2 if a.closure else None)
aco = ACOConfig(n_scouts=a.scouts)

if a.animate:
    from aco_traffic.animation import run_animation
    # run_animation(city, sim_cfg, a.router, aco)
    run_animation(
    city,
    sim_cfg,
    a.router,
    aco,
    scale=0.5,
    fps=30,
    steps_per_frame=1
    )
else:
    sim = Simulation(generate_city(city), sim_cfg,
                     lambda net, rng: make_router(a.router, net, rng, aco), seed=a.seed)
    sim.run()
    for k, v in compute_metrics(sim).items():
        print(f"{k:22s} {v:.3f}" if isinstance(v, float) else f"{k:22s} {v}")
    if a.show:
        fig, ax = plt.subplots(figsize=(7, 6))
        pher = None
        if hasattr(sim.router, "pheromone_map"):
            pher = sim.router.pheromone_map(sim.nodes[-1])
        draw_city(sim.net, ax, pheromone=pher, title=f"{a.router}, t={sim.t:.0f}s")
        plt.show()

