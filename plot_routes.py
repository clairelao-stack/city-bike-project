from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import contextily as ctx

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

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    results_dir = root / 'results'
    data = json.loads((root / 'model_station_inputs.json').read_text())
    depots = json.loads((root / 'selected_depots.json').read_text())
    result = json.loads((results_dir / 'solution.json').read_text())
    assert result.get('has_feasible_solution'), 'A feasible saved solution is required.'
    plot_routes(data, depots, result, results_dir)
    print('Street map saved to', results_dir / 'routes.png')
    from IPython import get_ipython
    from IPython.display import Image, display
    if get_ipython() is not None:
        display(Image(filename=str(results_dir / 'routes.png')))
