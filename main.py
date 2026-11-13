import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from optimization_model import latlon_to_km, solve_citi_bike_rebalancing
from plot_routes import plot_routes


def validate_solution(data, depots, results_dir, capacity, max_distance, alpha, beta):
    import csv, json, math

    out9 = results_dir
    out9.mkdir(exist_ok=True)
    sol9 = json.loads((results_dir / 'solution.json').read_text())
    dep9 = depots
    st9 = data['stations']
    stations9 = {s['id']: s for s in st9}
    nodes9 = {**stations9, **{d['depot_id']: d for d in dep9}}
    results9 = {s['id']: s for s in sol9['station_results']}
    Q9, W9, alpha9, beta9 = capacity, max_distance, alpha, beta

    def distance9(a, b):
        return float(latlon_to_km(a['lat'], a['lon'], b['lat'], b['lon']))

    def csv9(name, rows):
        with (out9 / name).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    # Independently reconstruct loads, distance and inventories from exported routes.
    assert sol9['has_feasible_solution']
    assert len(stations9) == len(st9) == len(results9) == len(sol9['station_results'])
    assert set(stations9) == set(results9)
    assert set(sol9['routes']) == {d['truck_id'] for d in dep9}
    seen9, route_rows9 = set(), []
    delivery9 = dict.fromkeys(stations9, 0)
    total_distance9, moved9 = 0, 0
    for d in dep9:
        truck = d['truck_id']
        r = sol9['routes'][truck]
        path = r['path']
        assert path[0] == path[-1] == d['depot_id']
        assert set(r['transfers']) == set(path[1:-1])
        load, distance = 0, 0
        route_rows9.append({'truck':truck, 'order':0, 'station_id':path[0], 'name':d['name'],
                            'net_delivery':0, 'load_after':0, 'distance_so_far_km':0})
        for order, (a,b) in enumerate(zip(path[:-1], path[1:]), 1):
            distance += distance9(nodes9[a], nodes9[b])
            z = 0
            if b in stations9:
                assert b not in seen9
                seen9.add(b)
                z = r['transfers'][b]
                assert abs(z-round(z)) < 1e-6
                delivery9[b] += z
                load -= z  # Positive transfer delivers to a station and reduces truck load.
                assert math.isclose(load, r['loads'][b], abs_tol=1e-6)
                moved9 += max(-z, 0)
            else:
                assert b == d['depot_id'] and order == len(path)-1
            assert 0 <= load <= Q9
            route_rows9.append({'truck':truck, 'order':order, 'station_id':b, 'name':nodes9[b]['name'],
                               'net_delivery':z, 'load_after':load, 'distance_so_far_km':round(distance,3)})
        assert load == 0 and sum(r['transfers'].values()) == 0
        assert distance <= W9 + 1e-6
        assert math.isclose(distance, r['distance_km'], abs_tol=1e-6)
        total_distance9 += distance

    station_rows9 = []
    for sid, s in stations9.items():
        r = results9[sid]
        final = s['I0'] + delivery9[sid]
        assert 0 <= final <= s['C']
        assert r['I0'] == s['I0'] and r['target'] == s['I_star']
        assert r['net_delivery'] == delivery9[sid] and r['final_inventory'] == final
        assert r['mismatch'] == abs(final-s['I_star'])
        station_rows9.append({'station_id':sid, 'name':s['name'], 'effective_capacity':s['C'],
            'midnight_inventory':s['I0'], 'target_inventory':s['I_star'], 'net_delivery_total':delivery9[sid],
            'final_inventory':final, 'absolute_mismatch_before':abs(s['I0']-s['I_star']),
            'absolute_mismatch_after':abs(final-s['I_star']), 'visited':sid in seen9})

    initial9 = sum(s['I0'] for s in st9)
    target9 = sum(s['I_star'] for s in st9)
    before9 = sum(s['absolute_mismatch_before'] for s in station_rows9)
    after9 = sum(s['absolute_mismatch_after'] for s in station_rows9)
    lower9 = abs(target9-initial9)
    cost9 = alpha9*total_distance9 + beta9*after9
    assert sum(delivery9.values()) == 0
    assert sum(s['final_inventory'] for s in station_rows9) == initial9
    assert target9 == initial9
    assert after9 >= lower9
    assert before9 == sol9['baseline_mismatch'] and after9 == sol9['total_mismatch']
    assert math.isclose(total_distance9, sol9['total_distance_km'], abs_tol=1e-6)
    assert math.isclose(cost9, sol9['objective'], abs_tol=1e-6)
    assert math.isclose(beta9*before9, sol9['baseline_objective'], abs_tol=1e-6)
    assert cost9 <= beta9*before9 + 1e-6
    summary9 = {'station_count':len(st9), 'visited_station_count':len(seen9), 'bikes_moved':moved9,
        'initial_inventory_total':initial9, 'target_inventory_total':target9, 'conservation_mismatch_lower_bound':lower9,
        'mismatch_before':before9, 'mismatch_after':after9, 'mismatch_reduction':before9-after9,
        'approximate_total_distance_km':round(total_distance9,3), 'transport_cost':round(alpha9*total_distance9,4),
        'mismatch_penalty_cost':beta9*after9, 'objective':round(cost9,4),
        'no_rebalancing_objective':beta9*before9, 'objective_reduction':round(beta9*before9-cost9,4),
        'proven_optimal':sol9['status'] == 'Optimal', 'checks_passed':True}
    csv9('route_steps.csv', route_rows9)
    csv9('station_before_after.csv', station_rows9)
    csv9('summary.csv', [summary9])
    checks9 = [{'check':x, 'result':'passed'} for x in [
        'Inputs and results cover the same complete station set', 'Each truck starts and returns at its own depot',
        'Each station is visited at most once across the fleet', 'Integer transfers agree with reconstructed route loads',
        'Loads respect truck capacity and trucks start/finish empty', 'Each truck respects its approximate distance budget',
        'Final station inventory is nonnegative and within effective capacity', 'Regional bike total is conserved',
        'Targets and initial stock agree with inputs', 'Recomputed distances, mismatch and objective agree',
        'Mismatch respects the conservation lower bound', 'Objective is no worse than no rebalancing']]
    csv9('validation_checks.csv', checks9)
    (out9/'stage9_summary.json').write_text(json.dumps(summary9,ensure_ascii=False,indent=2))
    print(json.dumps(summary9, ensure_ascii=False, indent=2))
    print('\nAll 12 check categories passed. Positive transfer means delivery; negative means pickup.')
    for row in route_rows9:
        print(row['truck'], row['order'], row['name'], 'net_delivery_total', row['net_delivery'], 'load', row['load_after'])
    return pd.DataFrame(station_rows9), summary9


def main():
    parser = argparse.ArgumentParser(description="Solve the Midtown Citi Bike case.")
    parser.add_argument("-p", "--time-limit", type=float, default=600, help="Solver time limit in seconds")
    parser.add_argument("-Q", "--truck-capacity", type=int, default=25, help="Truck capacity (bikes)")
    parser.add_argument("-w", "--max-distance", type=float, default=25, help="Maximum route distance limit (km)")
    parser.add_argument("-a", "--alpha", type=float, default=1.5, help="Distance multiplier cost")
    parser.add_argument("-b", "--beta", type=float, default=5.0, help="Mismatch penalty cost")
    parser.add_argument("--warm-start", default=None, help="Optional saved route JSON")
    parser.add_argument("--heuristic-seconds", type=float, default=0,
                        help="Optional local-search time before solving (0 disables it)")
    args = parser.parse_args()

    if not all(math.isfinite(value) and value > 0 for value in (
        args.time_limit, args.truck_capacity, args.max_distance, args.alpha, args.beta
    )):
        parser.error("Time limit, capacity, max distance, alpha, and beta must be positive and finite.")

    if not math.isfinite(args.heuristic_seconds) or args.heuristic_seconds < 0:
        parser.error("Heuristic seconds must be finite and nonnegative.")
    if args.heuristic_seconds > 0 and not args.warm_start:
        parser.error("--heuristic-seconds needs a --warm-start route file.")

    ROOT = Path(__file__).resolve().parent
    # Prefer the case just generated by preparation/data_preparation.py.
    prepared_input = ROOT / 'outputs_stage6/model_station_inputs.json'
    station_input = prepared_input if prepared_input.is_file() else ROOT / 'model_station_inputs.json'
    if not station_input.is_file():
        raise FileNotFoundError(f'No station input in {ROOT}. Run preparation/data_preparation.py here first.')
    data = json.loads(station_input.read_text())
    print('Station inputs:', station_input)

    # A rebuilt station set needs depot selection for that same set.
    if prepared_input.is_file() or not (ROOT / 'selected_depots.json').is_file():
        from itertools import combinations
        stations = sorted(data['stations'], key=lambda station: station['id'])
        if len(stations) < 2:
            raise ValueError('At least two eligible stations are required to select two depots.')
        latitudes = np.array([station['lat'] for station in stations])
        longitudes = np.array([station['lon'] for station in stations])
        depot_distances = np.round(
            84 * np.abs(longitudes[:, None] - longitudes[None, :])
            + 111 * np.abs(latitudes[:, None] - latitudes[None, :]), 3
        )
        best_pair = min(
            combinations(range(len(stations)), 2),
            key=lambda pair: np.minimum(depot_distances[:, pair[0]], depot_distances[:, pair[1]]).sum()
        )
        nearest = np.argmin(depot_distances[:, list(best_pair)], axis=1)
        depots = []
        for index, station_index in enumerate(best_pair):
            station = stations[station_index]
            truck = f'Truck_{index + 1}'
            depots.append({
                'truck_id': truck, 'depot_id': f'depot_{truck}',
                'name': f"Depot {index + 1} (candidate near {station['name']})",
                'lat': station['lat'], 'lon': station['lon'],
                'candidate_station_id': station['id'],
                'nearest_station_count': int((nearest == index).sum()),
                'site_feasibility_verified': False,
            })
        depot_output = ROOT / 'outputs_stage7'
        depot_output.mkdir(exist_ok=True)
        (depot_output / 'selected_depots.json').write_text(json.dumps(depots, indent=2))
    else:
        depots = json.loads((ROOT / 'selected_depots.json').read_text())

    RESULTS = ROOT / 'results'
    RESULTS.mkdir(exist_ok=True)
    station_table = pd.DataFrame(data['stations'])
    assert station_table['id'].is_unique
    assert (station_table['G'] == station_table['I_star'] - station_table['I0']).all()
    assert station_table['I_star'].sum() == station_table['I0'].sum()
    print(station_table[['id', 'name', 'C', 'I0', 'I_star', 'G']].head(10))
    print(f"Stations: {len(station_table)}; trucks: {len(depots)}; initial bikes: {station_table['I0'].sum()}")

    alpha, beta = args.alpha, args.beta
    solution = solve_citi_bike_rebalancing(
        data, depots, RESULTS, str(ROOT / "gurobi.lic"),
        truck_capacity=args.truck_capacity, max_distance_w=args.max_distance,
        alpha=alpha, beta=beta, time_limit=args.time_limit,
        warm_start_path=(ROOT / args.warm_start) if args.warm_start else None,
        heuristic_seconds=args.heuristic_seconds,
    )
    (RESULTS / "solution.json").write_text(json.dumps(solution, indent=2))
    _, summary = validate_solution(
        data, depots, RESULTS, args.truck_capacity, args.max_distance, alpha, beta,
    )
    plot_routes(data, depots, solution, RESULTS)
    reduction = summary['mismatch_before'] - summary['mismatch_after']
    print(f"Inventory deviation fell from {summary['mismatch_before']} to {summary['mismatch_after']} bikes.")
    print(f"Reduction: {reduction} bikes; total approximate travel: {solution['total_distance_km']:.3f} km.")
    print(f"Solver status: {solution['status']}; objective: {solution['objective']:.3f}; gap: {solution['gap']:.2%}.")
    print('Outputs saved in', RESULTS)


if __name__ == "__main__":
    main()