"""Generator losowego miasta: regiony (dzielnice), ulice lokalne i nieliczne drogi główne"""
import networkx as nx
import numpy as np

from .config import CityConfig


def _add_road(G, u, v, kind, cfg, rng):
    (x1, y1), (x2, y2) = G.nodes[u]["pos"], G.nodes[v]["pos"]
    length = float(np.hypot(x2 - x1, y2 - y1))
    if kind == "main":
        speed, lanes = cfg.main_speed, cfg.main_lanes
    else:
        speed, lanes = float(rng.uniform(*cfg.local_speed_range)), cfg.local_lanes
    capacity = max(3.0, lanes * length / cfg.veh_spacing)   # liczba pojazdów obsługiwanych przez krawędź
    for a, b in ((u, v), (v, u)):
        G.add_edge(a, b, length=length, free_speed=speed, capacity=capacity,
                   t0=length / speed, kind=kind)


def _connect_region(G, ids, cfg, rng):
    """
    Ulice lokalne: k najbliższych sąsiadów + dołączenie brakujących składowych
    """
    pts = np.array([G.nodes[i]["pos"] for i in ids])
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    H = nx.Graph()
    H.add_nodes_from(range(len(ids)))
    for a in range(len(ids)):
        for b in np.argsort(d[a])[1:cfg.knn + 1]:
            H.add_edge(a, int(b))
    while not nx.is_connected(H):
        comp = nx.node_connected_component(H, 0)
        _, a, b = min((d[a, b], a, b) for a in comp for b in range(len(ids)) if b not in comp)
        H.add_edge(a, b)
    for a, b in H.edges:
        _add_road(G, ids[a], ids[b], "local", cfg, rng)


def _link_regions(G, A, B, cfg, rng):
    """Drogi główne między dwoma sąsiednimi regionami (wąskie gardło sieci)."""
    pairs = sorted((float(np.hypot(*np.subtract(G.nodes[a]["pos"], G.nodes[b]["pos"]))), a, b)
                   for a in A for b in B)
    used, n = set(), 0
    for _, a, b in pairs:
        if a in used or b in used:
            continue
        _add_road(G, a, b, "main", cfg, rng)
        used |= {a, b}
        n += 1
        if n >= cfg.links_per_adjacent_pair:
            break


def generate_city(cfg: CityConfig) -> nx.DiGraph:
    """
    Skierowany graf ważony: wierzchołki = skrzyżowania, krawędzie = odcinki ulic.

    Atrybuty krawędzi: length [m], free_speed [m/s], capacity [pojazdy], t0 [s], kind.
    Atrybuty wierzchołków: pos (x, y), region. G.graph["regions"] = {region: [węzły]}.
    """
    rng = np.random.default_rng(cfg.seed)
    G = nx.DiGraph()
    regions, nid = {}, 0
    for ry in range(cfg.n_regions_y):
        for rx in range(cfg.n_regions_x):
            r = ry * cfg.n_regions_x + rx
            origin = np.array([rx, ry]) * cfg.region_size
            pts = rng.uniform(0.08, 0.92, size=(cfg.nodes_per_region, 2)) * cfg.region_size + origin
            ids = []
            for p in pts:
                G.add_node(nid, pos=(float(p[0]), float(p[1])), region=r)
                ids.append(nid)
                nid += 1
            regions[r] = ids
            _connect_region(G, ids, cfg, rng)
    for ry in range(cfg.n_regions_y):
        for rx in range(cfg.n_regions_x):
            r = ry * cfg.n_regions_x + rx
            if rx + 1 < cfg.n_regions_x:
                _link_regions(G, regions[r], regions[r + 1], cfg, rng)
            if ry + 1 < cfg.n_regions_y:
                _link_regions(G, regions[r], regions[r + cfg.n_regions_x], cfg, rng)
    G.graph["regions"] = regions
    return G


def pick_closure_edge(G):
    """
    Droga główna o największej pośredniczości (przy swobodnym ruchu) -
    ta sama dla wszystkich algorytmów, więc porównanie jest uczciwe
    """
    bc = nx.edge_betweenness_centrality(G, weight="t0")
    main = [e for e in G.edges if G.edges[e]["kind"] == "main"]
    return max(main, key=lambda e: (bc[e], e))