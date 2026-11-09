"""Model ruchu (funkcja BPR), samochody i symulator w dyskretnych krokach czasowych."""
from dataclasses import dataclass, field
from time import perf_counter as clock
from typing import List, Optional, Tuple

import numpy as np

from .city import pick_closure_edge
from .config import SimConfig


def bpr_time(t0: float, load: float, capacity: float, a: float = 0.15, b: float = 4.0) -> float:
    """Funkcja BPR: t = t0 * (1 + 0.15 * (v/c)^4)."""
    return t0 * (1.0 + a * (load / capacity) ** b)


class Network:
    """Stan sieci: obciążenia krawędzi, zamknięte drogi i aktualne czasy przejazdu."""

    def __init__(self, G):
        self.G = G
        self.load = {e: 0 for e in G.edges}
        self.closed = set()
        self.pos = {n: G.nodes[n]["pos"] for n in G.nodes}
        self.region = {n: G.nodes[n]["region"] for n in G.nodes}
        self.regions = G.graph["regions"]
        self.succ = {n: list(G.successors(n)) for n in G.nodes}
        self.vmax = max(d["free_speed"] for _, _, d in G.edges(data=True))

    # funkcje wag dla NetworkX: weight(u, v, dane) -> liczba lub None (krawędź niedostępna)
    def w_current(self, u, v, d):
        if (u, v) in self.closed:
            return None
        return bpr_time(d["t0"], self.load[(u, v)], d["capacity"])

    def w_free(self, u, v, d):
        return d["t0"]

    def w_free_open(self, u, v, d):
        return None if (u, v) in self.closed else d["t0"]

    def travel_time(self, u, v) -> float:
        d = self.G[u][v]
        return bpr_time(d["t0"], self.load[(u, v)], d["capacity"])

    def open_neighbors(self, u):
        if not self.closed:
            return self.succ[u]
        return [v for v in self.succ[u] if (u, v) not in self.closed]

    def dist(self, a, b) -> float:
        (x1, y1), (x2, y2) = self.pos[a], self.pos[b]
        return float(np.hypot(x2 - x1, y2 - y1))

    def close_road(self, u, v):
        self.closed.add((u, v))
        if self.G.has_edge(v, u):
            self.closed.add((v, u))


@dataclass
class Car:
    id: int
    origin: int
    dest: int
    dest_region: int
    spawn_t: float
    node: int
    path: List[Tuple[int, float]] = field(default_factory=list)  # (węzeł, czas dotarcia)
    prev: int = -1
    edge: Optional[Tuple[int, int]] = None
    remaining: float = 0.0
    total: float = 0.0
    route: Optional[list] = None      # pozostałe węzły trasy (routery Dijkstra/A*)
    hops: int = 0
    arrive_t: Optional[float] = None


class Simulation:
    """Symulacja w krokach dt. Kolejność w kroku: (zamknięcie) -> pojawianie się aut ->
    decyzje na skrzyżowaniach -> ruch po krawędziach -> aktualizacja routera -> statystyki."""

    def __init__(self, G, cfg: SimConfig, router_factory, seed: int = 0):
        self.cfg = cfg
        self.net = Network(G)
        self.rng = np.random.default_rng(seed + 1000)                  # popyt
        self.router = router_factory(self.net, np.random.default_rng(seed + 2000))
        self.nodes = list(G.nodes)
        self.edges = list(G.edges)
        self.caps = np.array([G.edges[e]["capacity"] for e in self.edges], dtype=float)
        self.cars, self.waiting, self.moving = [], [], []
        self.t, self.step_no, self.next_id = 0.0, 0, 0
        self.closure_edge = pick_closure_edge(G) if cfg.closure_time is not None else None
        self.closed_done = False
        self.router_time, self.n_decisions = 0.0, 0
        self.util_sum = self.util_over = self.util_max = 0.0
        self.util_n = 0

    def _pick_dest(self, origin):
        net, r = self.net, self.net.region[origin]
        others = [x for x in net.regions if x != r]
        if others and self.rng.random() < self.cfg.p_inter_region:
            r2 = int(self.rng.choice(others))
            return int(self.rng.choice(net.regions[r2]))
        cand = [n for n in net.regions[r] if n != origin]
        return int(self.rng.choice(cand))

    def _spawn(self):
        for _ in range(self.rng.poisson(self.cfg.rate * self.cfg.dt)):
            o = int(self.rng.choice(self.nodes))
            d = self._pick_dest(o)
            car = Car(self.next_id, o, d, self.net.region[d], self.t, o, path=[(o, self.t)])
            self.next_id += 1
            t0 = clock()
            self.router.on_spawn(car, self.t)
            self.router_time += clock() - t0
            self.cars.append(car)
            self.waiting.append(car)

    def step(self):
        cfg, net, t = self.cfg, self.net, self.t
        if self.closure_edge and not self.closed_done and t >= cfg.closure_time:
            net.close_road(*self.closure_edge)
            self.router.on_closure()
            self.closed_done = True
        self._spawn()

        waiting, self.waiting = self.waiting, []
        for car in waiting:
            t0 = clock()
            nxt = self.router.next_node(car, t)
            self.router_time += clock() - t0
            self.n_decisions += 1
            if nxt is None:                      # brak dostępnej drogi - czeka
                self.waiting.append(car)
                continue
            e = (car.node, nxt)
            net.load[e] += 1
            d = net.G[car.node][nxt]
            car.edge = e
            car.remaining = car.total = bpr_time(d["t0"], net.load[e], d["capacity"])
            self.moving.append(car)

        still = []
        for car in self.moving:
            car.remaining -= cfg.dt
            if car.remaining > 0:
                still.append(car)
                continue
            u, v = car.edge
            net.load[car.edge] -= 1
            car.prev, car.node, car.edge = u, v, None
            car.path.append((v, t + cfg.dt))
            if v == car.dest:
                car.arrive_t = t + cfg.dt
                t0 = clock()
                self.router.on_arrive(car, car.arrive_t)
                self.router_time += clock() - t0
            else:
                self.waiting.append(car)
        self.moving = still

        t0 = clock()
        self.router.on_step(self.step_no, t)
        self.router_time += clock() - t0

        if t >= cfg.warmup:
            u = np.fromiter((net.load[e] for e in self.edges), float, len(self.edges)) / self.caps
            self.util_sum += float(u.mean())
            self.util_over += float((u > 1.0).mean())
            self.util_max = max(self.util_max, float(u.max()))
            self.util_n += 1
        self.t += cfg.dt
        self.step_no += 1

    def run(self):
        while self.t < self.cfg.duration:
            self.step()
        return self