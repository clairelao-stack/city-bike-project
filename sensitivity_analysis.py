"""
Citi Bike Rebalancing Model - Sensitivity Analysis Pipeline
===========================================================
Executes One-At-A-Time (OFAT) parameter sweeps for Q, w, alpha, and beta,
logs model performance and stats to Weights & Biases (W&B), and saves
local CSV, JSON, and 4-panel visual dashboard artifacts.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import solver and validation functions directly from project modules
from main import validate_solution
from optimization_model import solve_citi_bike_rebalancing
from plot_routes import plot_routes

# Optional Weights & Biases integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# Baseline settings
DEFAULT_PARAMS = {
    'truck_capacity': 25,
    'max_distance': 25.0,
    'alpha': 1.5,
    'beta': 5.0,
}

# Parameter sweep grids
PARAM_SWEEPS = {
    'Q (Truck Capacity)': {
        'param_key': 'truck_capacity',
        'values': [20, 25, 30, 35, 40],
    },
    'w (Max Distance km)': {
        'param_key': 'max_distance',
        'values': [20.0, 25.0, 30.0, 35.0, 40.0],
    },
    'alpha (Transport Weight)': {
        'param_key': 'alpha',
        'values': [1.3, 1.4, 1.5, 1.6, 1.7],
    },
    'beta (Mismatch Penalty)': {
        'param_key': 'beta',
        'values': [4.0, 4.5, 5.0, 5.5, 6.0],
    },
}


def load_inputs(root_dir: Path):
    """Loads input station data and depot configuration files."""
    prepared_input = root_dir / 'outputs_stage6' / 'model_station_inputs.json'
    station_input = prepared_input if prepared_input.is_file() else root_dir / 'model_station_inputs.json'
    if not station_input.is_file():
        raise FileNotFoundError(f'Station input file not found in {root_dir}.')
    data = json.loads(station_input.read_text())

    depot_file = root_dir / 'outputs_stage7' / 'selected_depots.json'
    if not depot_file.is_file():
        depot_file = root_dir / 'selected_depots.json'
    if not depot_file.is_file():
        raise FileNotFoundError(f'Depot configuration file not found in {root_dir}.')
    depots = json.loads(depot_file.read_text())

    return data, depots


def run_single_experiment(data, depots, root_dir, out_base_dir, time_limit, Q, w, a, b, sweep_name, val):
    """Runs solver for a single parameter configuration and returns recorded statistics."""
    run_id = f"sweep_{sweep_name}_Q{Q}_w{w}_a{a}_b{b}".replace('.', '_').replace(' ', '_').replace('(', '').replace(')', '')
    run_dir = out_base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    license_path = str(root_dir / "gurobi.lic")

    start_time = time.time()
    solution = solve_citi_bike_rebalancing(
        data, depots, run_dir, license_path,
        truck_capacity=int(Q), max_distance_w=float(w),
        alpha=float(a), beta=float(b), time_limit=time_limit,
    )
    (run_dir / "solution.json").write_text(json.dumps(solution, indent=2))

    _, summary = validate_solution(
        data, depots, run_dir, int(Q), float(w), float(a), float(b),
    )
    elapsed_total = time.time() - start_time

    try:
        plot_routes(data, depots, solution, run_dir)
    except Exception as err:
        print(f"Warning: Route plotting failed for {run_id}: {err}")

    record = {
        'run_id': run_id,
        'swept_parameter': sweep_name,
        'swept_value': val,
        'truck_capacity_Q': int(Q),
        'max_distance_w': float(w),
        'alpha': float(a),
        'beta': float(b),
        'time_limit_p': time_limit,
        'objective': float(summary['objective']),
        'total_distance_km': float(summary['approximate_total_distance_km']),
        'transport_cost': float(summary['transport_cost']),
        'mismatch_penalty_cost': float(summary['mismatch_penalty_cost']),
        'mismatch_before': int(summary['mismatch_before']),
        'mismatch_after': int(summary['mismatch_after']),
        'mismatch_reduction': int(summary['mismatch_reduction']),
        'bikes_moved': int(summary['bikes_moved']),
        'visited_stations': int(summary['visited_station_count']),
        'no_rebalancing_objective': float(summary['no_rebalancing_objective']),
        'objective_reduction': float(summary['objective_reduction']),
        'solver_status': solution['status'],
        'mip_gap': float(solution.get('gap', 0.0)),
        'solver_runtime_sec': float(solution.get('runtime_seconds', elapsed_total)),
    }
    return record, run_dir


def generate_local_dashboard(df: pd.DataFrame, output_path: Path):
    """Generates a 4-panel dashboard summarizing parameter sensitivity."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Citi Bike Rebalancing Sensitivity Analysis Dashboard", fontsize=18, fontweight='bold', y=0.98)

    # 1. Total Objective Cost vs Parameter Values
    ax1 = axes[0, 0]
    sns.lineplot(data=df, x='swept_value', y='objective', hue='swept_parameter', marker='o', linewidth=2.5, ax=ax1)
    ax1.set_title("Total Objective Cost ($Z = \\alpha D + \\beta U$)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Parameter Value", fontsize=11)
    ax1.set_ylabel("Objective Value ($)", fontsize=11)

    # 2. Total Travel Distance (km) vs Parameter Values
    ax2 = axes[0, 1]
    sns.lineplot(data=df, x='swept_value', y='total_distance_km', hue='swept_parameter', marker='s', linewidth=2.5, ax=ax2)
    ax2.set_title("Total Travel Distance (km)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Parameter Value", fontsize=11)
    ax2.set_ylabel("Distance (km)", fontsize=11)

    # 3. Post-Rebalancing Target Mismatch (Bikes)
    ax3 = axes[1, 0]
    sns.lineplot(data=df, x='swept_value', y='mismatch_after', hue='swept_parameter', marker='^', linewidth=2.5, ax=ax3)
    ax3.set_title("Remaining Inventory Mismatch (Bikes)", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Parameter Value", fontsize=11)
    ax3.set_ylabel("Mismatch Count", fontsize=11)

    # 4. Transport Cost vs Penalty Cost Breakdown
    ax4 = axes[1, 1]
    df_melted = df.melt(
        id_vars=['swept_parameter', 'swept_value'], 
        value_vars=['transport_cost', 'mismatch_penalty_cost'],
        var_name='Cost Type', value_name='Cost ($)'
    )
    df_melted['Cost Type'] = df_melted['Cost Type'].map({
        'transport_cost': 'Transport Cost (alpha * dist)',
        'mismatch_penalty_cost': 'Penalty Cost (beta * mismatch)'
    })
    sns.barplot(data=df_melted, x='swept_parameter', y='Cost ($)', hue='Cost Type', ax=ax4, errorbar=None)
    ax4.set_title("Cost Component Breakdown by Parameter Sweep", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Swept Parameter Group", fontsize=11)
    ax4.set_ylabel("Cost ($)", fontsize=11)
    ax4.tick_params(axis='x', rotation=15)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Local dashboard plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Sensitivity Analysis for Citi Bike Rebalancing Model.")
    parser.add_argument("-p", "--time-limit", type=float, default=2000, help="Solver time limit per run in seconds")
    parser.add_argument("--wandb-project", type=str, default="citi-bike-sensitivity", help="W&B Project Name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B Entity/Username (optional)")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parent
    out_base_dir = ROOT / "results_sensitivity"
    out_base_dir.mkdir(exist_ok=True)

    data, depots = load_inputs(ROOT)
    print(f"Loaded {len(data['stations'])} stations and {len(depots)} depots.")
    print(f"Time limit per run: {args.time_limit} seconds.")

    results_list = []
    runs_to_execute = []
    executed_configs = {}

    for param_name, config in PARAM_SWEEPS.items():
        key = config['param_key']
        for val in config['values']:
            params = DEFAULT_PARAMS.copy()
            params[key] = val
            config_tuple = (params['truck_capacity'], params['max_distance'], params['alpha'], params['beta'])

            runs_to_execute.append({
                'sweep_name': param_name,
                'swept_value': val,
                'Q': params['truck_capacity'],
                'w': params['max_distance'],
                'a': params['alpha'],
                'b': params['beta'],
                'config_tuple': config_tuple
            })

    total_runs = len(runs_to_execute)
    print(f"Total sensitivity runs scheduled: {total_runs}")

    for idx, run_info in enumerate(runs_to_execute, 1):
        print(f"\n[{idx}/{total_runs}] Running Sweep: {run_info['sweep_name']} = {run_info['swept_value']}")
        print(f"Config -> Q={run_info['Q']}, w={run_info['w']}, alpha={run_info['a']}, beta={run_info['b']}")

        config_tuple = run_info['config_tuple']
        if config_tuple in executed_configs:
            print("Configuration already executed. Reusing solution stats...")
            base_rec, run_dir = executed_configs[config_tuple]
            rec = base_rec.copy()
            rec['swept_parameter'] = run_info['sweep_name']
            rec['swept_value'] = run_info['swept_value']
            rec['run_id'] = f"sweep_{run_info['sweep_name']}_val_{run_info['swept_value']}".replace('.', '_').replace(' ', '_')
        else:
            rec, run_dir = run_single_experiment(
                data, depots, ROOT, out_base_dir, args.time_limit,
                run_info['Q'], run_info['w'], run_info['a'], run_info['b'],
                run_info['sweep_name'], run_info['swept_value']
            )
            executed_configs[config_tuple] = (rec, run_dir)

        results_list.append(rec)

        # Log individual run to W&B
        if WANDB_AVAILABLE and not args.disable_wandb:
            try:
                run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=f"{run_info['sweep_name']}={run_info['swept_value']}",
                    config={
                        'swept_parameter': run_info['sweep_name'],
                        'swept_value': run_info['swept_value'],
                        'truck_capacity': run_info['Q'],
                        'max_distance': run_info['w'],
                        'alpha': run_info['a'],
                        'beta': run_info['b'],
                        'time_limit': args.time_limit
                    },
                    reinit=True
                )
                wandb.log(rec)
                run.finish()
            except Exception as err:
                print(f"W&B logging skipped/failed: {err}")

    # Export aggregated tabular data
    df_results = pd.DataFrame(results_list)
    csv_path = out_base_dir / "sensitivity_results.csv"
    json_path = out_base_dir / "sensitivity_results.json"
    df_results.to_csv(csv_path, index=False)
    json_path.write_text(df_results.to_json(orient="records", indent=2))
    print(f"\nAll sensitivity statistics exported to:")
    print(f"  - CSV:  {csv_path}")
    print(f"  - JSON: {json_path}")

    # Render local dashboard plot
    dashboard_path = out_base_dir / "sensitivity_analysis_dashboard.png"
    generate_local_dashboard(df_results, dashboard_path)

    # Master W&B Dashboard Summary Upload
    if WANDB_AVAILABLE and not args.disable_wandb:
        try:
            master_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name="Sensitivity_Summary_Dashboard",
                job_type="analysis",
                reinit=True
            )
            wandb.log({"Sensitivity_Dashboard": wandb.Image(str(dashboard_path))})
            wandb.log({"Sensitivity_Table": wandb.Table(dataframe=df_results)})
            master_run.finish()
            print("Uploaded summary table and visualization dashboard to W&B.")
        except Exception as err:
            print(f"Master W&B upload skipped: {err}")


if __name__ == "__main__":
    main()