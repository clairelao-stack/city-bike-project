"""只在这里设置路径和实验参数，然后运行这个文件。"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

# 你主要修改下面四项。
WARM_START_FILE = ROOT / 'warm_start_83457.json'
HEURISTIC_SECONDS = 30  # 改成 0，就是只读 warm start、不运行新增启发式。
SOLVER_SECONDS = 600
OUTPUT_FOLDER = ROOT / ('results_heuristic' if HEURISTIC_SECONDS else 'results_warm_only')

# 先保持原模型的容量和距离预算，方便比较。
TRUCK_CAPACITY = 25
MAX_DISTANCE = 25

if __name__ == '__main__':
    command = [sys.executable, str(ROOT/'main.py'),
               '--warm-start', str(WARM_START_FILE),
               '--heuristic-seconds', str(HEURISTIC_SECONDS),
               '--output-dir', str(OUTPUT_FOLDER),
               '-p', str(SOLVER_SECONDS), '-Q', str(TRUCK_CAPACITY),
               '--max-distance', str(MAX_DISTANCE)]
    subprocess.run(command, check=True)
