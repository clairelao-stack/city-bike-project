"""只读取并改进初始方案，不启动 Gurobi。"""
import argparse
import json
import math
from pathlib import Path

from warm_start import prepare_warm_start

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warm-start', type=Path, default=ROOT/'warm_start_83457.json')
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'prepared_start')
    args = parser.parse_args()
    if args.seconds < 0 or not math.isfinite(args.seconds):
        parser.error('Seconds must be finite and nonnegative.')
    data = json.loads((ROOT/'model_station_inputs.json').read_text())
    depots = json.loads((ROOT/'selected_depots.json').read_text())
    distances = {}
    for depot in depots:
        points = {s['id']: (s['lat'], s['lon']) for s in data['stations']}
        points[depot['depot_id']] = (depot['lat'], depot['lon'])
        for i, a in points.items():
            for j, b in points.items():
                if i != j:
                    distances[i, j, depot['truck_id']] = round(
                        84*abs(a[1]-b[1])+111*abs(a[0]-b[0]), 3)
    start_path = args.warm_start
    if not start_path.is_absolute():
        start_path = ROOT/start_path
    output = args.output_dir
    if not output.is_absolute():
        output = ROOT/output
    prepare_warm_start(start_path, data, depots, distances,
                       25, 25, 1.5, 5, args.seconds, output)
    print('Prepared route saved. No MIP was solved.')
