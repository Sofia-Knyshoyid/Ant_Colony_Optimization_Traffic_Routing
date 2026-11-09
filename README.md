# Routing mrówkowy (ACO) w mieście z korkami

Symulator miasta z ograniczoną przepustowością dróg i porównanie ACO
(hierarchiczne i płaskie tablice feromonowe, opcjonalnie mrówki zwiadowcze)
z Dijkstrą i A* (warianty statyczne i dynamiczne).

## Uruchomienie

    pip install -r requirements.txt
    pytest                                   # szybkie testy
    python run_demo.py --router aco_hier --traffic peak --show

    # eksperymenty (wyniki i wykresy w katalogu results/)
    python -m aco_traffic.experiments traffic   --seeds 5 --jobs 4
    python -m aco_traffic.experiments closure   --seeds 5 --level medium
    python -m aco_traffic.experiments city_size --seeds 5
    python -m aco_traffic.experiments params    --seeds 5 --level peak
    python -m aco_traffic.experiments hier_flat --seeds 5

## Routery
static_dijkstra, static_astar, dynamic_dijkstra, dynamic_astar, aco_hier, aco_flat
(liczbę mrówek zwiadowczych ustawia ACOConfig.n_scouts).

## Uwagi
- Ziarno (seed) steruje miastem, popytem i decyzjami routera niezależnie,
  więc wszystkie algorytmy dostają ten sam popyt (porównanie sparowane).
- Pierwsze ~300 s symulacji to rozgrzewka (uczenie feromonu); metryki liczone po niej.
- Stench pheromone i predykcja czasów przejazdu nie są częścią podstawowej wersji.
