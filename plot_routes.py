from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pyproj import Transformer


def plot_routes(data, depots, result, results_dir):
    stations = {s['id']: s for s in data['stations']}
    nodes = {**stations, **{d['depot_id']: d for d in depots}}
    project = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    points = {key: project.transform(node['lon'], node['lat'])
              for key, node in nodes.items()}
    fig, ax = plt.subplots(figsize=(19, 17))
    for sign, color, label in [
        (-1, '#244f70', 'Surplus: pickup (G < 0)'),
        (1, '#cf3e28', 'Deficit: deliver (G > 0)'),
        (0, '#999999', 'Balanced (G = 0)'),
    ]:
        positions = [points[key] for key, station in stations.items()
                     if (station['G'] > 0) - (station['G'] < 0) == sign]
        ax.scatter([x for x, _ in positions], [y for _, y in positions],
                   s=27, c=color, edgecolors='white', linewidths=.4, label=label, zorder=4)

    labels = []
    colors = ['#29906b', '#85499f']
    for index, depot in enumerate(depots):
        truck = depot['truck_id']
        route = result['routes'][truck]
        path = route['path']
        color = colors[index % len(colors)]
        stops = [key for key in path if key in stations]
        ax.plot([points[key][0] for key in path], [points[key][1] for key in path],
                color=color, lw=1.6, alpha=.85, zorder=3,
                label=f"{truck}: {len(stops)} stops, {route['distance_km']:.3f} km")
        for start, end in zip(path, path[1:]):
            x, y = points[start]
            dx, dy = points[end][0] - x, points[end][1] - y
            ax.annotate('', xy=(x + .72*dx, y + .72*dy), xytext=(x + .49*dx, y + .49*dy),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.4), zorder=3)
        for order, key in enumerate(stops, 1):
            station = stations[key]
            transfer = route['transfers'].get(key, 0)
            labels.append((points[key], f"{order}. "
                           f"G={station['G']}, trans={transfer:+.0f}", color))
        x, y = points[depot['depot_id']]
        ax.scatter(x, y, marker='*', s=370, color='gold', edgecolors='black',
                   zorder=7, label=f'{truck} depot')
        ax.annotate(f'Depot {index + 1}', (x, y), xytext=(0, -15), textcoords='offset points',
                    ha='center', va='top', fontsize=8, zorder=8,
                    bbox=dict(facecolor='gold', edgecolor='black', pad=3))

    ax.set_aspect('equal')
    ax.margins(.10)
    ax.set(xlabel='Web Mercator easting (m)', ylabel='Web Mercator northing (m)',
           title='Connecting lines show truck visit order')
    fig.suptitle('Truck visit order — Midtown Manhattan (2026-10-28)', fontsize=20, fontweight='bold', y=.97)
    handles, legend_labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], linestyle='none'))
    legend_labels.append(f"Total absolute target mismatch: {result['total_mismatch']} bikes")
    legend = ax.legend(handles, legend_labels, loc='upper left')
    try:
        import contextily as ctx

        # The axes and tiles already use Web Mercator.
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldStreetMap,
                        zoom=14, alpha=.65, zorder=0)
    except Exception as error:
        print(f'Street basemap unavailable ({error}). Saving the route map without street tiles.')
        ax.grid(alpha=.18, zorder=0)

    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = [legend.get_window_extent(renderer)]
    offsets = [(x, y) for y in (8, -26, 26, -43, 44, -60, 62)
               for x in (8, -8, 42, -42, 80, -80, 120, -120)]
    for point, label, color in labels:
        note = ax.annotate(label, point, xytext=offsets[0], textcoords='offset points',
                           fontsize=6, color=color, ha='left', va='bottom', zorder=6,
                           bbox=dict(facecolor='white', edgecolor=color, alpha=.91, pad=2),
                           arrowprops=dict(arrowstyle='-', color=color, lw=.5, alpha=.65))
        for x, y in offsets:
            note.set_position((x, y))
            note.set_ha('right' if x < 0 else 'left')
            note.update_positions(renderer)
            note.update_bbox_position_size(renderer)
            box = note.get_bbox_patch().get_window_extent(renderer).expanded(1.03, 1.12)
            if (ax.bbox.contains(*box.min) and ax.bbox.contains(*box.max)
                    and not any(box.overlaps(other) for other in occupied)):
                break
        occupied.append(box)
    fig.savefig(Path(results_dir) / 'routes.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    results_dir = root / 'results'
    station_input = root / 'outputs_stage6/model_station_inputs.json'
    depot_input = root / 'outputs_stage7/selected_depots.json'
    if not station_input.is_file():
        station_input = root / 'model_station_inputs.json'
        depot_input = root / 'selected_depots.json'
    data = json.loads(station_input.read_text())
    depots = json.loads(depot_input.read_text())
    result = json.loads((results_dir / 'solution.json').read_text())
    assert result.get('has_feasible_solution'), 'A feasible saved solution is required.'
    plot_routes(data, depots, result, results_dir)
    print('Route map saved to', results_dir / 'routes.png')
