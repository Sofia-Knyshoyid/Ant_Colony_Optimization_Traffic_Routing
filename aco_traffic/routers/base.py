"""Wspólny interfejs routerów"""
import tracemalloc


class Router:
    name = "base"

    def __init__(self, net, rng):
        self.net = net
        self.rng = rng

    def on_spawn(self, car, t):
        pass

    def next_node(self, car, t):
        """Zwraca następne skrzyżowanie (sąsiada car.node) albo None"""
        raise NotImplementedError

    def on_arrive(self, car, t):
        pass

    def on_step(self, step_no, t):
        pass

    def on_closure(self):
        pass

    def storage_entries(self) -> int:
        """Liczba wpisów pamięci routera (tablice feromonowe); 0 dla Dijkstry/A*"""
        return 0

    def _sample_decision(self):
        raise NotImplementedError

    def decision_memory_bytes(self) -> int:
        """Szczytowe zużycie pamięci roboczej pojedynczej decyzji routingowej (tracemalloc)"""
        was = tracemalloc.is_tracing()
        if not was:
            tracemalloc.start()
        tracemalloc.reset_peak()
        base, _ = tracemalloc.get_traced_memory()
        self._sample_decision()
        _, peak = tracemalloc.get_traced_memory()
        if not was:
            tracemalloc.stop()
        return max(0, peak - base)