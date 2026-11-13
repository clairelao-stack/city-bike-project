import argparse
import math
parser = argparse.ArgumentParser(description='Solve Citi Bike rebalancing and save plots.')
parser.add_argument('-p', '--time-limit', type=float, default=600, help='Solver time limit in seconds')
parser.add_argument('-Q', '--truck-capacity', type=int, default=25)
parser.add_argument('--max-distance', type=float, default=25)
parser.add_argument('--warm-start', default=None, help='Saved solution JSON')
parser.add_argument('--heuristic-seconds', type=float, default=0, help='Optional local search before the MIP')
parser.add_argument('--output-dir', default=None, help='Separate results directory')
args = parser.parse_args()
if not math.isfinite(args.time_limit) or args.time_limit <= 0 or args.truck_capacity <= 0 or not math.isfinite(args.max_distance) or args.max_distance <= 0:
    parser.error('Time limit, capacity and distance must be positive and finite.')

from warm_start import prepare_warm_start, apply_warm_start
from pathlib import Path
import json
import csv
import math
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import contextily as ctx
from IPython.display import display
import gurobipy as gp
from gurobipy import GRB

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / 'model_station_inputs.json').read_text())
depots = json.loads((ROOT / 'selected_depots.json').read_text())
RESULTS = Path(args.output_dir) if args.output_dir else ROOT / 'results'
if not RESULTS.is_absolute():
    RESULTS = ROOT / RESULTS
RESULTS.mkdir(parents=True, exist_ok=True)
station_table = pd.DataFrame(data['stations'])
assert station_table['id'].is_unique
assert (station_table['G'] == station_table['I_star'] - station_table['I0']).all()
assert station_table['I_star'].sum() == station_table['I0'].sum()
display(station_table[['id', 'name', 'C', 'I0', 'I_star', 'G']].head(10))
print(f"Stations: {len(station_table)}; trucks: {len(depots)}; initial bikes: {station_table['I0'].sum()}")


truck_capacity = args.truck_capacity
max_distance_w = args.max_distance
alpha = 1.5
beta = 5
time_limit = args.time_limit
license_path = str(ROOT / 'gurobi.lic')

def latlon_to_km(lat1, lon1, lat2, lon2):
    return np.round(84 * np.abs(lon1 - lon2) + 111 * np.abs(lat1 - lat2), 3)

def init_gurobi_env(license_path: str = "gurobi.lic") -> gp.Env:
    """Initializes Gurobi Environment using local gurobi.lic license file."""
    abs_lic_path = os.path.abspath(license_path)
    if os.path.exists(abs_lic_path) and not os.environ.get("GRB_LICENSE_FILE"):
        os.environ["GRB_LICENSE_FILE"] = abs_lic_path
        print(f"Loading Gurobi license from: {abs_lic_path}")
    
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.start()
    return env

all_stations = data['stations']
stations = all_stations
I = [s['id'] for s in stations]
K = [d['truck_id'] for d in depots]

C = {s['id']: s['C'] for s in stations}
I0 = {s['id']: s['I0'] for s in stations}
G = {s['id']: s['G'] for s in stations}

depot_map = {d['truck_id']: d for d in depots}

# Distance Matrix D_ij
D = {}
for k in K:
    nodes = I + [f"depot_{k}"]
    coords = {s['id']: (s['lat'], s['lon']) for s in stations}
    coords[f"depot_{k}"] = (depot_map[k]['lat'], depot_map[k]['lon'])

    for i in nodes:
        for j in nodes:
            if i != j:
                lat1, lon1 = coords[i]
                lat2, lon2 = coords[j]
                D[(i, j, k)] = latlon_to_km(lat1, lon1, lat2, lon2)

max_capacity = max(C.values()) if len(C) > 0 else 50
M = 2 * truck_capacity + max_capacity

env = init_gurobi_env(license_path=license_path)
model = gp.Model("CitiBike_Midtown_Rebalancing", env=env)
model.Params.TimeLimit = time_limit
model.Params.Threads = 2



# Decision Variables
x = {}  # Routing variable x_{ijk} in {0,1}
z = {}  # Net delivery z_{ik} in Integer
l = {}  # Load l_{ik} >= 0
u = {}  # Absolute target mismatch u_i >= 0
y = {}  # Subtour elimination sequence variable y_{ik} >= 0

for i in I:
    u[i] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"u_{i}")

for k in K:
    depot_k = f"depot_{k}"
    nodes_k = I + [depot_k]

    l[(depot_k, k)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=0, name=f"l_{depot_k}_{k}")

    for i in I:
        z[(i, k)] = model.addVar(vtype=GRB.INTEGER, lb=-truck_capacity, ub=truck_capacity, name=f"z_{i}_{k}")
        l[(i, k)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=truck_capacity, name=f"l_{i}_{k}")
        y[(i, k)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=len(I), name=f"y_{i}_{k}")

    for i in nodes_k:
        for j in nodes_k:
            if i != j:
                x[(i, j, k)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

model.update()



# Objective Function (Equation 3)
total_distance = gp.quicksum(D[arc] * x[arc] for arc in x)
transport_cost = alpha * total_distance
penalty_cost = gp.quicksum(beta * u[i] for i in I)
model.setObjective(transport_cost + penalty_cost, GRB.MINIMIZE)



# (4) Depot Flow Balance
for k in K:
    depot_k = f"depot_{k}"
    model.addConstr(
        gp.quicksum(x[(depot_k, j, k)] for j in I) == gp.quicksum(x[(i, depot_k, k)] for i in I),
        name=f"depot_flow_balance_{k}"
    )
    model.addConstr(
        gp.quicksum(x[(depot_k, j, k)] for j in I) <= 1,
        name=f"depot_use_{k}"
    )

# (5) Route Continuity
for k in K:
    depot_k = f"depot_{k}"
    nodes_k = I + [depot_k]
    for i in nodes_k:
        model.addConstr(
            gp.quicksum(x[(j, i, k)] for j in nodes_k if j != i) == 
            gp.quicksum(x[(i, j, k)] for j in nodes_k if j != i),
            name=f"continuity_{i}_{k}"
        )

# (6) At most one visit per station
for i in I:
    model.addConstr(
        gp.quicksum(x[(j, i, k)] for k in K for j in I + [f"depot_{k}"] if j != i) <= 1,
        name=f"single_visit_{i}"
    )



# (7) Transfers bounded by visit
for k in K:
    depot_k = f"depot_{k}"
    nodes_k = I + [depot_k]
    for i in I:
        visit_var = gp.quicksum(x[(j, i, k)] for j in nodes_k if j != i)
        model.addConstr(z[(i, k)] >= -I0[i] * visit_var, name=f"transfer_lb_{i}_{k}")
        model.addConstr(z[(i, k)] <= (C[i] - I0[i]) * visit_var, name=f"transfer_ub_{i}_{k}")

# (8) Station Dock Capacity Constraints
for i in I:
    total_transfer = gp.quicksum(z[(i, k)] for k in K)
    model.addConstr(I0[i] + total_transfer >= 0, name=f"station_min_inv_{i}")
    model.addConstr(I0[i] + total_transfer <= C[i], name=f"station_max_inv_{i}")

# (9) Absolute Target Mismatch Penalty
for i in I:
    total_transfer = gp.quicksum(z[(i, k)] for k in K)
    model.addConstr(G[i] - total_transfer <= u[i], name=f"penalty_pos_{i}")
    model.addConstr(G[i] - total_transfer >= -u[i], name=f"penalty_neg_{i}")



# (10) Vehicle Load Bound
for k in K:
    depot_k = f"depot_{k}"
    nodes_k = I + [depot_k]
    for i in I:
        visit_var = gp.quicksum(x[(j, i, k)] for j in nodes_k if j != i)
        model.addConstr(l[(i, k)] <= truck_capacity * visit_var, name=f"load_ub_{i}_{k}")

# (11) Big-M Load Flow Balance
for k in K:
    depot_k = f"depot_{k}"
    nodes_k = I + [depot_k]
    for j in I:
        for i in nodes_k:
            if i != j:
                model.addConstr(
                    l[(j, k)] - l[(i, k)] + z[(j, k)] >= -M * (1 - x[(i, j, k)]),
                    name=f"load_track_lb_{i}_{j}_{k}"
                )
                model.addConstr(
                    l[(j, k)] - l[(i, k)] + z[(j, k)] <= M * (1 - x[(i, j, k)]),
                    name=f"load_track_ub_{i}_{j}_{k}"
                )

# (12) Empty Return to Depot
for k in K:
    depot_k = f"depot_{k}"
    for i in I:
        model.addConstr(
            l[(i, k)] <= truck_capacity * (1 - x[(i, depot_k, k)]),
            name=f"empty_return_{i}_{k}"
        )



# (13) Max Travel Budget Limit
for k in K:
    depot_k = f"depot_{k}"
    nodes_k = I + [depot_k]
    model.addConstr(
        gp.quicksum(D[(i, j, k)] * x[(i, j, k)] for i in nodes_k for j in nodes_k if i != j) <= max_distance_w,
        name=f"dist_limit_{k}"
    )

# (14) SUBTOUR ELIMINATION (MTZ Sequence Ordering Constraints)
for k in K:
    depot_k = f"depot_{k}"
    for i in I:
        visit_var = gp.quicksum(x[(j, i, k)] for j in I + [depot_k] if j != i)
        model.addConstr(y[(i, k)] <= len(I) * visit_var, name=f"seq_ub_{i}_{k}")
        model.addConstr(y[(i, k)] >= 1 - len(I) * (1 - x[(depot_k, i, k)]), name=f"seq_depot_{i}_{k}")
        for j in I:
            if i != j:
                model.addConstr(
                    y[(j, k)] >= y[(i, k)] + 1 - (len(I)+1) * (1 - x[(i, j, k)]),
                    name=f"seq_step_{i}_{j}_{k}"
                )


for k in K:
    model.addConstr(gp.quicksum(z[i,k] for i in I)==0,name=f'explicit_conservation_{k}')



# 初始方案的读取、检查和可选改进交给独立文件；原模型约束不动。
if not math.isfinite(args.heuristic_seconds) or args.heuristic_seconds < 0:
    raise ValueError('Heuristic seconds must be finite and nonnegative.')
if args.heuristic_seconds > 0 and not args.warm_start:
    raise ValueError('Local search needs a --warm-start file.')
if args.warm_start:
    start_path = Path(args.warm_start)
    if not start_path.is_absolute():
        start_path = ROOT / start_path
    start = prepare_warm_start(start_path, data, depots, D, truck_capacity,
                              max_distance_w, alpha, beta, args.heuristic_seconds, RESULTS)
    apply_warm_start(start, x, z, l, u, y, I, K, G)

model.Params.LogFile = str(RESULTS / 'gurobi.log')
model.optimize()
if model.SolCount == 0:
    raise RuntimeError('No feasible solution was found. Inspect the solver log.')


def extract_solution(model, data, depots, x, z, l, D, alpha, beta):
    all_stations = data['stations']
    I = [station['id'] for station in all_stations]
    K = [depot['truck_id'] for depot in depots]
    G = {station['id']: station['G'] for station in all_stations}
    status_labels = {
        GRB.OPTIMAL: 'Optimal',
        GRB.TIME_LIMIT: 'Time limit reached',
        GRB.INFEASIBLE: 'Infeasible',
    }
    status = status_labels.get(model.Status, str(model.Status))
    runtime=model.Runtime
    if model.SolCount == 0:
        raise RuntimeError('Gurobi returned no feasible incumbent. Check the log; do not reuse old results.')
    gap=model.MIPGap if I else 0.0
    bound=model.ObjBound
    result={'solver':'Gurobi','status':status,'runtime_seconds':runtime,'has_feasible_solution':True,
            'gap':gap,'objective_bound':bound,'stations':len(all_stations),'routing_stations':len(I),'trucks':len(K),'routes':{},'station_results':[]}
    distance=0.0
    for k in K:
        depot=f'depot_{k}'
        nodes=I+[depot]
        arcs = []
        for i in nodes:
            for j in nodes:
                if i != j and x[i, j, k].X > 0.5:
                    arcs.append((i, j))
        next_stop = dict(arcs)
        path = [depot]
        current = depot
        for _ in range(len(I)+2):
            if current not in next_stop:
                break
            current = next_stop[current]
            path.append(current)
            if current == depot:
                break
        assert not arcs or (path[-1]==depot and len(path)-1==len(arcs))
        dist=sum(D[i,j,k] for i,j in arcs)
        distance+=dist
        result['routes'][k]={'path':path,'distance_km':dist,
            'transfers':{i:round(z[i, k].X) for i in I if i in path},
            'loads':{i:l[i, k].X for i in I if i in path}}
    for station in all_stations:
        i = station['id']
        target_gap = station['G']
        net = round(sum(z[i, k].X for k in K))
        initial = station['I0']
        final = initial + net
        assert 0 <= final <= station['C']
        result['station_results'].append({
            'id':i, 'I0':initial, 'net_delivery':net, 'final_inventory':final,
            'target':station['I_star'], 'mismatch':abs(target_gap-net),
        })
    assert sum(r['net_delivery'] for r in result['station_results'])==0
    mismatch=sum(r['mismatch'] for r in result['station_results'])
    result.update(total_distance_km=distance,total_mismatch=mismatch,objective=alpha*distance+beta*mismatch,
        baseline_mismatch=sum(abs(v) for v in G.values()),baseline_objective=beta*sum(abs(v) for v in G.values()))
    return result

solution = extract_solution(model, data, depots, x, z, l, D, alpha, beta)
solution['parameters'] = {'truck_capacity': truck_capacity, 'max_distance': max_distance_w, 'time_limit': time_limit, 'alpha': alpha, 'beta': beta}
(RESULTS / 'solution.json').write_text(json.dumps(solution, indent=2))
model.dispose()
env.dispose()
display({key: value for key, value in solution.items() if key not in ['routes', 'station_results']})


def validate_solution(capacity, max_distance, alpha, beta):
    import csv, json, math
    from pathlib import Path
    import matplotlib.pyplot as plt

    out9 = RESULTS
    out9.mkdir(exist_ok=True)
    sol9 = json.loads((RESULTS / 'solution.json').read_text())
    dep9 = json.loads((ROOT / 'selected_depots.json').read_text())
    st9 = json.loads((ROOT / 'model_station_inputs.json').read_text())['stations']
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


before_after, summary = validate_solution(truck_capacity, max_distance_w, alpha, beta)
display(before_after)


def plot_routes(data, depots, result, results_dir):
    stations = {s['id']: s for s in data['stations']}
    nodes = {**stations, **{d['depot_id']: d for d in depots}}
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter([s['lon'] for s in stations.values()], [s['lat'] for s in stations.values()], s=12, color='gray', alpha=0.6)
    for depot in depots:
        truck = depot['truck_id']
        path = result['routes'][truck]['path']
        ax.plot([nodes[i]['lon'] for i in path], [nodes[i]['lat'] for i in path], marker='o', markersize=3, label=truck)
        ax.scatter(depot['lon'], depot['lat'], marker='*', s=220, color='gold', edgecolors='black', zorder=5)
        for order, sid in enumerate(path[1:-1], 1):
            ax.annotate(str(order), (nodes[sid]['lon'], nodes[sid]['lat']), fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.set_aspect(1 / np.cos(np.deg2rad(np.mean([s['lat'] for s in stations.values()]))))
    ax.set(xlabel='Longitude', ylabel='Latitude', title='Truck visit order — Midtown Manhattan')
    ax.legend()
    print('Downloading street basemap...')
    try:
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.Esri.WorldStreetMap,
                        zoom=14, alpha=0.75, attribution_size=7, zorder=0)
    except Exception as error:
        plt.close(fig)
        raise RuntimeError('Street map download failed. Check internet access and rerun plot_routes.py.') from error
    fig.tight_layout()
    fig.savefig(Path(results_dir) / 'routes.png', dpi=180)
    plt.close(fig)

plot_routes(data, depots, solution, RESULTS)


reduction = summary['mismatch_before'] - summary['mismatch_after']
print(f"Inventory deviation fell from {summary['mismatch_before']} to {summary['mismatch_after']} bikes.")
print(f"Reduction: {reduction} bikes; total approximate travel: {solution['total_distance_km']:.3f} km.")
print(f"Solver status: {solution['status']}; objective: {solution['objective']:.3f}; gap: {solution['gap']:.2%}.")
print('Outputs saved in', RESULTS)

from IPython import get_ipython
from IPython.display import Image
if get_ipython() is not None:
    display(Image(filename=str(RESULTS / 'routes.png')))
