"""Warm start 读取、可行性检查、局部搜索与 .Start 设置集中在本文件。

运行示例（在项目目录的 Jupyter 中）：
%run main.py -p 600 --warm-start warm_start_route.json --heuristic-seconds 30

--heuristic-seconds 0：只用保存路线，不运行新增局部搜索。
原 optimization_model.py 的数学约束、参数和回调均保留。
这是自定义局部搜索，不是 Clarke-Wright。
"""
from collections import deque
from copy import deepcopy
import random
import json
from pathlib import Path
from time import perf_counter


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


def refresh_start(start, data, depots, distances, capacity, budget, alpha, beta):
    """重算成本和载重，只保留会影响路线可行性的判断。"""
    stations = {s['id']: s for s in data['stations']}
    routes = {}
    visited = set()
    for depot in depots:
        truck = depot['truck_id']
        route = start['routes'][truck]
        path = route['path']
        stops = path[1:-1]
        if len(stops) != len(set(stops)) or visited & set(stops):
            raise ValueError('A station is visited more than once.')
        visited.update(stops)
        load = 0
        loads = {}
        for i in stops:
            transfer = route['transfers'][i]
            load -= transfer
            if not (0 <= load <= capacity and 0 <= stations[i]['I0']+transfer <= stations[i]['C']):
                raise ValueError('The saved route does not fit the current capacity or inventory.')
            loads[i] = load
        km = route_distance(path, distances, truck)
        if load != 0 or km > budget+1e-6:
            raise ValueError('The saved route must return empty and respect the distance budget.')
        routes[truck] = dict(path=path[:], transfers=route['transfers'].copy(),
                             loads=loads, distance_km=km)
    return summarize_start(data['stations'], routes, capacity, budget, alpha, beta)


def prepare_warm_start(path, data, depots, distances, capacity, budget, alpha, beta,
                       heuristic_seconds, output_dir, seed=42):
    with Path(path).open(encoding='utf-8') as file:
        start = json.load(file)
    started = perf_counter()
    start = refresh_start(start, data, depots, distances, capacity, budget, alpha, beta)
    before = start['objective']
    if heuristic_seconds > 0:
        start, _ = improve_warm_start(
            start, data, depots, distances, capacity, budget, alpha, beta,
            seconds=heuristic_seconds, seed=seed)
    # 原项目 main.py 会在求解后继续执行完整的结果验证。
    output_dir = Path(output_dir)
    report = dict(original_objective=before, prepared_objective=start['objective'],
                  preparation_seconds=perf_counter()-started)
    (output_dir/'prepared_warm_start.json').write_text(json.dumps(start, indent=2))
    (output_dir/'warm_start_report.json').write_text(json.dumps(report, indent=2))
    print(f"Warm start: {before:.4f} -> {start['objective']:.4f}")
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


def plan_transfers(stops, gaps, capacity):
    """按顺序取送车，剩下没送出的车不取，确保空车返回。"""
    pending = deque()
    transfers = dict.fromkeys(stops, 0)
    load = 0
    for i in stops:
        if gaps[i] < 0:
            pickup = min(-gaps[i], capacity-load)
            if pickup > 0:
                pending.append((i, pickup))
                load += pickup
        else:
            delivered = min(gaps[i], load)
            transfers[i] = delivered
            load -= delivered
            while delivered > 0:
                source, available = pending.popleft()
                used = min(delivered, available)
                transfers[source] -= used
                delivered -= used
                if used < available:
                    pending.appendleft((source, available-used))
    loads = {}
    load = 0
    for i in stops:
        load -= transfers[i]
        loads[i] = load
    return transfers, loads


def evaluate_routes(stops_by_truck, data, depots, distances, capacity, budget, alpha, beta):
    gaps = {s['id']: s['G'] for s in data['stations']}
    seen = set()
    routes = {}
    for depot in depots:
        truck, depot_id = depot['truck_id'], depot['depot_id']
        stops = stops_by_truck[truck]
        if len(stops) != len(set(stops)) or seen & set(stops) or not set(stops) <= gaps.keys():
            return None
        seen.update(stops)
        path = [depot_id, *stops, depot_id] if stops else [depot_id]
        km = route_distance(path, distances, truck)
        if km > budget+1e-6:
            return None
        transfers, loads = plan_transfers(stops, gaps, capacity)
        routes[truck] = dict(path=path, transfers=transfers, loads=loads, distance_km=km)
    return summarize_start(data['stations'], routes, capacity, budget, alpha, beta)


def improve_warm_start(start, data, depots, distances, capacity, budget, alpha, beta,
                       seconds=30, seed=42):
    """从已有好路线开始；时间预算是启发式时间，不含后面的 MIP。"""
    rng = random.Random(seed)
    best = deepcopy(start)
    trucks = [d['truck_id'] for d in depots]
    ids = [s['id'] for s in data['stations']]
    gaps = {s['id']: s['G'] for s in data['stations']}
    begin = perf_counter()
    iterations = improvements = 0
    # 先尝试保持访问顺序、重新计算取送量；更差就不采用。
    stops = {k: r['path'][1:-1] for k, r in best['routes'].items()}
    candidate = evaluate_routes(stops, data, depots, distances, capacity, budget, alpha, beta)
    if candidate and candidate['objective'] < best['objective']-1e-6:
        best = candidate
        improvements += 1
    while perf_counter()-begin < seconds:
        iterations += 1
        stops = {k: r['path'][1:-1].copy() for k, r in best['routes'].items()}
        k = rng.choice(trucks)
        route = stops[k]
        visited = {i for r in stops.values() for i in r}
        unvisited = [i for i in ids if i not in visited]
        operation = rng.randrange(6)

        if operation == 0:  # 移动一个站，可以移到另一辆车。
            if not route:
                continue
            i = route.pop(rng.randrange(len(route)))
            other = rng.choice(trucks)
            stops[other].insert(rng.randrange(len(stops[other])+1), i)
        elif operation == 1:  # 交换两个站的访问位置。
            other = rng.choice(trucks)
            if not route or not stops[other]:
                continue
            a, b = rng.randrange(len(route)), rng.randrange(len(stops[other]))
            route[a], stops[other][b] = stops[other][b], route[a]
        elif operation == 2:  # 反转一小段访问顺序。
            if len(route) < 2:
                continue
            a, b = sorted(rng.sample(range(len(route)), 2))
            route[a:b+1] = reversed(route[a:b+1])
        elif operation == 3:  # 删除一个站。
            if not route:
                continue
            route.pop(rng.randrange(len(route)))
        elif operation == 4:  # 加入一个尚未访问的站。
            if not unvisited:
                continue
            route.insert(rng.randrange(len(route)+1), rng.choice(unvisited))
        else:  # 一起插入一个取车站和一个送车站，先取后送。
            pickups = [i for i in unvisited if gaps[i] < 0]
            deliveries = [i for i in unvisited if gaps[i] > 0]
            if not pickups or not deliveries:
                continue
            a = rng.randrange(len(route)+1)
            route.insert(a, rng.choice(pickups))
            b = rng.randrange(a+1, len(route)+1)
            route.insert(b, rng.choice(deliveries))

        candidate = evaluate_routes(stops, data, depots, distances, capacity, budget, alpha, beta)
        if candidate and candidate['objective'] < best['objective']-1e-6:
            best = candidate
            improvements += 1
    return best, dict(iterations=iterations, improvements=improvements,
                     seconds=perf_counter()-begin, seed=seed)
