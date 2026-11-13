"""读取、检查、保存初始方案，并把它传入原模型；不增加约束。"""
import json
import math
from pathlib import Path
from time import perf_counter


def read_warm_start(path):
    with Path(path).open(encoding='utf-8') as file:
        start = json.load(file)
    if not start.get('has_feasible_solution'):
        raise ValueError('The warm-start file must contain a feasible saved solution.')
    return start


def route_distance(path, distances, truck):
    return sum(distances[a, b, truck] for a, b in zip(path, path[1:]) if a != b)


def summarize_start(stations, routes, capacity, budget, alpha, beta):
    """由路线和取送量重算库存与成本，不沿用旧文件里的成本。"""
    net = {station['id']: 0 for station in stations}
    distance = 0
    for route in routes.values():
        distance += route['distance_km']
        for station, transfer in route['transfers'].items():
            net[station] += transfer
    station_results = []
    for station in stations:
        i = station['id']
        station_results.append(dict(
            id=i, I0=station['I0'], target=station['I_star'], net_delivery=net[i],
            final_inventory=station['I0'] + net[i], mismatch=abs(station['G']-net[i])))
    mismatch = sum(row['mismatch'] for row in station_results)
    baseline = sum(abs(station['G']) for station in stations)
    return dict(
        has_feasible_solution=True, solver='Saved start / local search',
        status='Feasible warm start', gap=None, objective_bound=None,
        stations=len(stations), routing_stations=len(stations), trucks=len(routes),
        routes=routes, station_results=station_results,
        total_distance_km=distance, total_mismatch=mismatch,
        objective=alpha*distance+beta*mismatch,
        baseline_mismatch=baseline, baseline_objective=beta*baseline,
        parameters=dict(truck_capacity=capacity, max_distance=budget, alpha=alpha, beta=beta))


def check_warm_start(start, data, depots, distances, capacity, budget, alpha, beta):
    """独立重放路线：检查原模型需要的库存、载重、访问和距离规则。"""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(capacity > 0 and math.isfinite(budget) and budget > 0, 'Invalid capacity or budget.')
    stations = {station['id']: station for station in data['stations']}
    require(len(stations) == len(data['stations']), 'Duplicate input station IDs.')
    trucks = {depot['truck_id'] for depot in depots}
    require(set(start['routes']) == trucks, 'Warm-start trucks differ from the input trucks.')
    old_stations = {row['id']: row for row in start.get('station_results', [])}
    require(set(old_stations) == set(stations), 'Warm-start station IDs differ from the inputs.')
    for i, station in stations.items():
        require(station['G'] == station['I_star']-station['I0'], f'Invalid target gap: {i}')
        old = old_stations[i]
        require(old['I0'] == station['I0'] and old['target'] == station['I_star'],
                'Inventory inputs changed; build a compatible initial plan first.')
    require(sum(s['I0'] for s in stations.values()) == sum(s['I_star'] for s in stations.values()),
            'This experiment expects fleet-conserving targets.')
    seen = set()
    clean_routes = {}
    for depot in depots:
        truck, depot_id = depot['truck_id'], depot['depot_id']
        route = start['routes'][truck]
        path = route['path']
        require(bool(path) and path[0] == path[-1] == depot_id,
                f'{truck}: route must start and end at its own depot.')
        stops = path[1:-1]
        require(len(stops) == len(set(stops)), f'{truck}: repeated station.')
        require(not (seen & set(stops)), 'Two trucks visit the same station.')
        require(set(stops) <= stations.keys(), f'{truck}: unknown station.')
        require(set(route['transfers']) == set(stops), f'{truck}: missing or extra transfers.')
        seen.update(stops)
        transfers, loads = {}, {}
        load = 0
        for i in stops:
            value = route['transfers'][i]
            require(math.isfinite(value) and abs(value-round(value)) < 1e-6,
                    f'{truck}: transfer must be an integer at {i}.')
            z = int(round(value))
            require(abs(z) <= capacity, f'{truck}: transfer exceeds truck capacity at {i}.')
            require(0 <= stations[i]['I0']+z <= stations[i]['C'], f'Station stock out of bounds: {i}')
            load -= z
            require(0 <= load <= capacity, f'{truck}: load out of bounds at {i}.')
            transfers[i], loads[i] = z, load
        require(load == 0, f'{truck}: truck must return empty.')
        km = route_distance(path, distances, truck)
        require(km <= budget+1e-6, f'{truck}: distance {km:.3f} exceeds budget {budget}.')
        clean_routes[truck] = dict(path=path[:], transfers=transfers, loads=loads, distance_km=km)
    return summarize_start(data['stations'], clean_routes, capacity, budget, alpha, beta)


def prepare_warm_start(path, data, depots, distances, capacity, budget, alpha, beta,
                       heuristic_seconds, output_dir, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    start = check_warm_start(read_warm_start(path), data, depots, distances,
                             capacity, budget, alpha, beta)
    before = start['objective']
    search_report = dict(iterations=0, improvements=0, seconds=0)
    if heuristic_seconds > 0:
        from heuristic import improve_warm_start
        start, search_report = improve_warm_start(
            start, data, depots, distances, capacity, budget, alpha, beta,
            seconds=heuristic_seconds, seed=seed)
    # 再检查一次；只有通过检查的方案才会交给原模型。
    start = check_warm_start(start, data, depots, distances, capacity, budget, alpha, beta)
    report = dict(source=str(Path(path).resolve()), original_objective=before,
                  prepared_objective=start['objective'], heuristic_enabled=heuristic_seconds > 0,
                  requested_heuristic_seconds=heuristic_seconds,
                  preparation_seconds=perf_counter()-started, search=search_report,
                  validation_passed=True)
    (output_dir/'prepared_warm_start.json').write_text(json.dumps(start, indent=2))
    (output_dir/'warm_start_report.json').write_text(json.dumps(report, indent=2))
    print(f"Warm start: {before:.4f} -> {start['objective']:.4f}; preparation {report['preparation_seconds']:.2f}s")
    return start


def apply_warm_start(start, x, z, load, mismatch, order, station_ids, truck_ids, gaps):
    """仅设置 .Start；不固定变量，也不增删原约束。"""
    for variables in (x, z, load, mismatch, order):
        for variable in variables.values():
            variable.Start = 0
    net = dict.fromkeys(station_ids, 0)
    for truck in truck_ids:
        route = start['routes'][truck]
        for a, b in zip(route['path'], route['path'][1:]):
            if a != b:
                x[a, b, truck].Start = 1
        for position, i in enumerate(route['path'][1:-1], 1):
            z[i, truck].Start = route['transfers'][i]
            load[i, truck].Start = route['loads'][i]
            order[i, truck].Start = position
            net[i] += route['transfers'][i]
    for i in station_ids:
        mismatch[i].Start = abs(gaps[i]-net[i])
