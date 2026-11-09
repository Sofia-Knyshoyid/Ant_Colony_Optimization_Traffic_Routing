import networkx as nx
import numpy as np
import pytest

from aco_traffic.city import generate_city
from aco_traffic.config import ACOConfig, CityConfig, SimConfig
from aco_traffic.metrics import compute_metrics
from aco_traffic.routers import ROUTER_NAMES, make_router
from aco_traffic.routers.aco import ACORouter
from aco_traffic.traffic import Network, Simulation, bpr_time


def test_city_reproducible_and_connected():
    G1, G2 = generate_city(CityConfig(seed=3)), generate_city(CityConfig(seed=3))
    assert list(G1.edges(data=True)) == list(G2.edges(data=True))
    assert nx.is_strongly_connected(G1)
    assert len(G1.graph["regions"]) == 9


def test_bpr():
    assert bpr_time(10, 0, 5) == 10
    assert bpr_time(10, 5, 5) == pytest.approx(11.5)


def test_loop_removal():
    G = generate_city(CityConfig(seed=1))
    r = ACORouter(Network(G), np.random.default_rng(0), ACOConfig())
    path = [(0, 0), (1, 1), (2, 2), (1, 3), (3, 4)]      # pętla 1 -> 2 -> 1
    r._reinforce(path, 4.0, dest=3)
    owner, key = r._row_owner(1, 3)
    assert 3 in owner[key] and 2 not in owner[key]


@pytest.mark.parametrize("name", ROUTER_NAMES)
def test_routers_run(name):
    G = generate_city(CityConfig(seed=0))
    cfg = SimConfig(duration=240, warmup=60, traffic="low", closure_time=150)
    sim = Simulation(G, cfg, lambda net, rng: make_router(name, net, rng, ACOConfig()), seed=0)
    sim.run()
    m = compute_metrics(sim)
    assert m["throughput_per_min"] > 0
    assert any(c.arrive_t is not None for c in sim.cars)