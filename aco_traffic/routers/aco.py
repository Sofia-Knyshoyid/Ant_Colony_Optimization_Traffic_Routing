"""Routing mrówkowy (ACO) z hierarchicznymi tablicami feromonowymi na skrzyżowaniach.

Inspiracje: ACS (Dorigo & Gambardella 1997), AntNet (Di Caro & Dorigo 1998),
hierarchiczny routing rojowy (Tatomir & Rothkrantz 2006), AVCAS (Jabbarpour et al. 2014).

Tablice (rzadkie, tworzone leniwie) - dla każdego skrzyżowania i:
  regional[i][region_docelowy][sąsiad] -> tau   (cel w innym regionie)
  local[i][skrzyżowanie_docelowe][sąsiad] -> tau (cel w tym samym regionie)
  flat[i][skrzyżowanie_docelowe][sąsiad] -> tau  (wariant płaski - porównanie pamięci i jakości)
Brak wpisu = wartość początkowa tau0 (która jest też dolnym ograniczeniem po parowaniu).
"""
from .base import Router


class ACORouter(Router):
    def __init__(self, net, rng, cfg):
        super().__init__(net, rng)
        self.cfg = cfg
        self.name = "aco_hier" if cfg.hierarchical else "aco_flat"
        if cfg.n_scouts:
            self.name += f"_s{cfg.n_scouts}"
        nodes = list(net.G.nodes)
        self.regional = {n: {} for n in nodes}
        self.local = {n: {} for n in nodes}
        self.flat = {n: {} for n in nodes}
        self.nodes = nodes

    # tablice
    def _row_owner(self, node, dest):
        """Zwraca (słownik wierszy węzła, klucz celu) zależnie od hierarchii"""
        if self.cfg.hierarchical:
            rn, rd = self.net.region[node], self.net.region[dest]
            if rn != rd:
                return self.regional[node], rd
            return self.local[node], dest
        return self.flat[node], dest

    def storage_entries(self):
        return sum(len(row) for tabs in (self.regional, self.local, self.flat)
                   for per_node in tabs.values() for row in per_node.values())

    # decyzja
    def _eta(self, node, j, dest):
        """Heurystyka: aktualny czas przejazdu odcinka i/lub odległość od celu (w czasie)"""
        tt = self.net.travel_time(node, j)
        rem = self.net.dist(j, dest) / self.net.vmax
        h = self.cfg.heuristic
        if h == "time":
            return 1.0 / tt
        if h == "dist":
            return 1.0 / (rem + 1.0)
        return 1.0 / (tt + rem)

    def choose_next(self, node, prev, dest, rng):
        """Losowanie sąsiada z prawdopodobieństwem ~ tau^alpha * eta^beta."""
        nbrs = self.net.open_neighbors(node)
        if not nbrs:
            return None
        cand = [j for j in nbrs if j != prev] or list(nbrs)   # bez zawracania, jeśli są alternatywy
        owner, key = self._row_owner(node, dest)
        row = owner.get(key)
        a, b, tau0 = self.cfg.alpha, self.cfg.beta, self.cfg.tau0
        weights = []
        for j in cand:
            tau = row.get(j, tau0) if row else tau0
            weights.append((tau ** a) * (self._eta(node, j, dest) ** b))
        total = sum(weights)
        if total <= 0:
            return cand[int(rng.integers(len(cand)))]
        r, acc = rng.random() * total, 0.0
        for j, w in zip(cand, weights):
            acc += w
            if r <= acc:
                return j
        return cand[-1]

    def next_node(self, car, t):
        return self.choose_next(car.node, car.prev, car.dest, self.rng)

    #  feromon
    def _reinforce(self, path, t_end, dest):
        """Usuwa pętle z trasy i wzmacnia feromon: delta = Q / (czas od węzła do celu)."""
        stack, idx = [], {}
        for node, t in path:
            if node in idx:
                k = idx[node]
                for n, _ in stack[k + 1:]:
                    del idx[n]
                del stack[k + 1:]
                stack[k] = (node, t)
            else:
                idx[node] = len(stack)
                stack.append((node, t))
        cfg = self.cfg
        for k in range(len(stack) - 1):
            i, ti = stack[k]
            j = stack[k + 1][0]
            T = max(t_end - ti, 1.0)
            owner, key = self._row_owner(i, dest)
            row = owner.setdefault(key, {})
            row[j] = min(cfg.tau_max, row.get(j, cfg.tau0) + cfg.q / T)

    def on_arrive(self, car, t):
        self._reinforce(car.path, t, car.dest)

    def _evaporate(self):
        """tau <- (1 - rho) * tau, z dolnym ograniczeniem tau0 (wpisy sprowadzone do tau0 są usuwane)."""
        f, tau0 = 1.0 - self.cfg.rho, self.cfg.tau0
        for tabs in (self.regional, self.local, self.flat):
            for per_node in tabs.values():
                for key in list(per_node):
                    row = per_node[key]
                    for j in list(row):
                        v = row[j] * f
                        if v <= tau0:
                            del row[j]
                        else:
                            row[j] = v
                    if not row:
                        del per_node[key]



    def _scout(self, origin, dest):
        """Wirtualna mrówka: idzie po sieci wg tej samej reguły, nie zajmuje dróg;
        czas = suma bieżących czasów przejazdu odcinków."""
        node, prev, t = origin, -1, 0.0
        path = [(node, 0.0)]
        for _ in range(self.cfg.scout_max_hops):
            nxt = self.choose_next(node, prev, dest, self.rng)
            if nxt is None:
                return
            t += self.net.travel_time(node, nxt)
            prev, node = node, nxt
            path.append((node, t))
            if node == dest:
                self._reinforce(path, t, dest)
                return

    def on_step(self, step_no, t):
        for _ in range(self.cfg.n_scouts):
            o = self.nodes[int(self.rng.integers(len(self.nodes)))]
            d = self.nodes[int(self.rng.integers(len(self.nodes)))]
            if o != d:
                self._scout(o, d)
        if step_no % self.cfg.evap_interval == 0:
            self._evaporate()

    # narzędzia
    def pheromone_map(self, dest):
        """Siła feromonu każdej krawędzi (i->j) dla zadanego celu - do wizualizacji"""
        out = {}
        for i, j in self.net.G.edges:
            owner, key = self._row_owner(i, dest)
            row = owner.get(key)
            out[(i, j)] = row.get(j, 0.0) if row else 0.0
        return out

    def _sample_decision(self):
        self.choose_next(self.nodes[0], -1, self.nodes[-1], self.rng)