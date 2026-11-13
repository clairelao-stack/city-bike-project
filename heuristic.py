"""简单局部搜索：尝试小改动，只保留成本更低的可行初始方案。

这是新增的路线改进步骤，不是 Clarke-Wright。它不修改原 MIP 的任何约束。
"""
from collections import deque
from copy import deepcopy
import random
from time import perf_counter

from warm_start import route_distance, summarize_start


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
                       seconds=30, seed=42, max_iterations=None):
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
        if max_iterations is not None and iterations >= max_iterations:
            break
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
