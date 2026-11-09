"""Konfiguracja miasta, symulacji i algorytmu ACO."""
from dataclasses import dataclass
from typing import Optional, Tuple

# Natężenie ruchu: liczba nowych samochodów na sekundę symulacji (dla miasta 3x3 regiony).
# Wartości trzeba skalibrować: w trybie "peak" główne drogi powinny się zbliżać do przepustowości.
TRAFFIC_LEVELS = {"low": 1.5, "medium": 4.0, "peak": 8.0}


@dataclass(frozen=True)
class CityConfig:
    n_regions_x: int = 3
    n_regions_y: int = 3
    nodes_per_region: int = 12
    region_size: float = 1000.0          # bok regionu [m]
    knn: int = 3                         # liczba najbliższych sąsiadów przy budowie ulic w regionie
    links_per_adjacent_pair: int = 2     # liczba dróg głównych między sąsiednimi regionami
    local_speed_range: Tuple[float, float] = (8.3, 13.9)   # m/s (30-50 km/h)
    main_speed: float = 19.4             # m/s (~70 km/h)
    local_lanes: int = 1
    main_lanes: int = 1
    veh_spacing: float = 50.0            # efektywny odstęp [m na pojazd] -> przepustowość krawędzi
    seed: int = 0


@dataclass(frozen=True)
class SimConfig:
    dt: float = 1.0
    duration: float = 1800.0
    warmup: float = 300.0
    traffic: str = "medium"              # low/medium/ peak
    rate_scale: float = 1.0              # skalowanie natężenia (np. dla większych miast)
    p_inter_region: float = 0.7          # prawdopodobieństwo, że cel jest w innym regionie
    closure_time: Optional[float] = None  # moment zamknięcia drogi (None znaczy brak zamknięcia)

    @property
    def rate(self) -> float:
        return TRAFFIC_LEVELS[self.traffic] * self.rate_scale


@dataclass(frozen=True)
class ACOConfig:
    alpha: float = 1.0                   # wpływ feromonu
    beta: float = 2.0                    # wpływ heurystyki
    rho: float = 0.05                    # współczynnik parowania (na zdarzenie parowania)
    evap_interval: int = 5               # parowanie co tyle kroków symulacji
    q: float = 60.0                      # stała wzmocnienia: delta = q / czas_przejazdu
    tau0: float = 0.1                    # początkowa (i minimalna) wartość feromonu
    tau_max: float = 10.0
    n_scouts: int = 0                    # mrówki zwiadowcze na krok symulacji
    scout_max_hops: int = 150
    hierarchical: bool = True            # True: tablice regionalne+lokalne, False: płaskie
    heuristic: str = "time_dist"         # "time" | "dist" | "time_dist"
    