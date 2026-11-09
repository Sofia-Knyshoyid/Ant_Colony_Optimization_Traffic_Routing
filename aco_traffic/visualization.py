"""Mapa miasta z kolorami obciążenia, opcjonalnie grubość linii = intensywność feromonu"""
import math

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def draw_city(net, ax=None, pheromone=None, title=None, max_util=1.5):
    ax = ax or plt.gca()
    cmap, norm = plt.get_cmap("RdYlGn_r"), Normalize(0, max_util)
    pmax = max(pheromone.values()) if pheromone and max(pheromone.values()) > 0 else 1.0
    segs, cols, widths = [], [], []
    for (u, v), load in net.load.items():
        x1, y1 = net.pos[u]
        x2, y2 = net.pos[v]
        L = math.hypot(x2 - x1, y2 - y1) or 1.0
        ox, oy = (y2 - y1) / L * 10, -(x2 - x1) / L * 10    # przesunięcie, by oba kierunki były widoczne
        segs.append([(x1 + ox, y1 + oy), (x2 + ox, y2 + oy)])
        cols.append("black" if (u, v) in net.closed
                    else cmap(norm(load / net.G[u][v]["capacity"])))
        widths.append(0.8 + 4.0 * pheromone.get((u, v), 0.0) / pmax if pheromone else 1.6)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=widths))
    xs = [p[0] for p in net.pos.values()]
    ys = [p[1] for p in net.pos.values()]
    ax.set_xlim(min(xs) - 50, max(xs) + 50)
    ax.set_ylim(min(ys) - 50, max(ys) + 50)
    ax.set_aspect("equal")
    ax.set_title(title or "Obciążenie dróg (czarne = zamknięte)")
    plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="obciążenie / przepustowość")
    return ax


def compare_plot(agg, metric, path):
    """Wykres porównujący metody (średnia ± std po ziarnach)"""
    from .experiments import bar_plot
    bar_plot(agg, metric, path)