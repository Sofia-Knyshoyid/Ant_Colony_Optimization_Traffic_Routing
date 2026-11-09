from ..config import ACOConfig
from .aco import ACORouter
from .base import Router
from .shortest_path import ShortestPathRouter

ROUTER_NAMES = ["static_dijkstra", "static_astar", "dynamic_dijkstra", "dynamic_astar",
                "aco_hier", "aco_flat"]


def make_router(name: str, net, rng, aco_cfg: ACOConfig = None) -> Router:
    aco_cfg = aco_cfg or ACOConfig()
    if name.startswith(("static_", "dynamic_")):
        mode, algo = name.split("_")
        return ShortestPathRouter(net, rng, dynamic=(mode == "dynamic"), algo=algo)
    if name in ("aco_hier", "aco_flat"):
        from dataclasses import replace
        return ACORouter(net, rng, replace(aco_cfg, hierarchical=(name == "aco_hier")))
    raise ValueError(f"Nieznany router: {name}")
