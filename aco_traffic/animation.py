"""Opcjonalna animacja ruchu (pygame): kolory dróg = obciążenie, kropki = samochody"""
import math

from .city import generate_city
from .routers import make_router
from .traffic import Simulation


def run_animation(city_cfg, sim_cfg, router_name, aco_cfg=None, scale=0.35, fps=30,
                  steps_per_frame=2):
    import pygame

    G = generate_city(city_cfg)
    sim = Simulation(G, sim_cfg, lambda net, rng: make_router(router_name, net, rng, aco_cfg),
                     seed=city_cfg.seed)
    net = sim.net
    W = int(city_cfg.n_regions_x * city_cfg.region_size * scale) + 40
    H = int(city_cfg.n_regions_y * city_cfg.region_size * scale) + 40
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    P = lambda p: (int(p[0] * scale) + 20, int(p[1] * scale) + 20)

    def color(u, v):
        if (u, v) in net.closed:
            return (0, 0, 0)
        x = min(net.load[(u, v)] / net.G[u][v]["capacity"], 1.5) / 1.5
        return (int(255 * x), int(200 * (1 - x)), 40)

    running = True
    while running and sim.t < sim_cfg.duration:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
        for _ in range(steps_per_frame):
            sim.step()
        screen.fill((245, 245, 245))
        for (u, v) in net.load:
            pygame.draw.line(screen, color(u, v), P(net.pos[u]), P(net.pos[v]), 2)
        for car in sim.moving:
            u, v = car.edge
            f = 1 - car.remaining / max(car.total, 1e-9)
            x = net.pos[u][0] + (net.pos[v][0] - net.pos[u][0]) * f
            y = net.pos[u][1] + (net.pos[v][1] - net.pos[u][1]) * f
            # pygame.draw.circle(screen, (30, 60, 200), P((x, y)), 2)
            pygame.draw.circle(screen, (30, 60, 200), P((x, y)), 6)
        pygame.display.set_caption(f"{router_name}  t={sim.t:.0f}s  aut: {len(sim.moving)}")
        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()