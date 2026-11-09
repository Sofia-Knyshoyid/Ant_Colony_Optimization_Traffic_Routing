"""Referencyjne routery: Dijkstra / A* w wariantach statycznym i dynamicznym (NetworkX)"""
import networkx as nx

from .base import Router


class ShortestPathRouter(Router):
    """static  - trasa z czasów swobodnego przepływu, liczona raz (przeliczana tylko, gdy
                 następny odcinek jest zamknięty, bo samochód fizycznie nie może nim jechać);
       dynamic - trasa z aktualnych czasów przejazdu (BPR), przeliczana na skrzyżowaniach"""

    def __init__(self, net, rng, dynamic=False, algo="astar", replan_every=1):
        super().__init__(net, rng)
        assert algo in ("dijkstra", "astar")
        self.dynamic, self.algo, self.replan_every = dynamic, algo, replan_every
        self.name = f"{'dynamic' if dynamic else 'static'}_{algo}"
        self._cache = {}

    def _heuristic(self, u, v):
        # dopuszczalna: odległość euklidesowa / maksymalna prędkość swobodna <= t0
        return self.net.dist(u, v) / self.net.vmax

    def _plan(self, src, dst, weight):
        G = self.net.G
        try:
            if self.algo == "astar":
                return nx.astar_path(G, src, dst, heuristic=self._heuristic, weight=weight)
            return nx.dijkstra_path(G, src, dst, weight=weight)
        except nx.NetworkXNoPath:
            return None

    def on_spawn(self, car, t):
        if self.dynamic:
            car.route = None
            return
        key = (car.origin, car.dest)
        if key not in self._cache:
            self._cache[key] = self._plan(car.origin, car.dest, self.net.w_free)
        p = self._cache[key]
        car.route = list(p[1:]) if p else None

    def next_node(self, car, t):
        net = self.net
        blocked = bool(car.route) and (car.node, car.route[0]) in net.closed
        car.hops += 1
        if self.dynamic:
            need = (not car.route) or blocked or (car.hops - 1) % self.replan_every == 0
            weight = net.w_current
        else:
            need = (not car.route) or blocked
            weight = net.w_free_open
        if need:
            p = self._plan(car.node, car.dest, weight)
            if p is None or len(p) < 2:
                return None
            car.route = list(p[1:])
        return car.route.pop(0)

    def _sample_decision(self):
        nodes = list(self.net.G.nodes)
        weight = self.net.w_current if self.dynamic else self.net.w_free
        self._plan(nodes[0], nodes[-1], weight)

