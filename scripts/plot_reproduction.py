"""Plot one persisted prediction and its reconstructed-paper enhancement; no synthetic layouts."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PatchPolygon
from shapely.affinity import affine_transform
from shapely.geometry import Polygon
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduce.verify import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    read = lambda name: json.loads((args.directory / name).read_text(encoding='utf-8'))
    data, summary = read('input.json'), read('summary.json')
    index = summary['selected_candidate']
    if index is None:
        index = 0
    raw = read('raw.json')[index]
    enhanced = read(f'candidate_{index:04d}.json')['actions']
    target = args.directory / 'comparison.png'
    if target.exists():
        raise FileExistsError(target)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    widths = []
    for ax, poses, label in zip(axes, (raw, enhanced), ('Model output', 'After enhancement')):
        metrics = measure(data['contours'], poses)
        shapes = []
        for contour, (x, y, c, s) in zip(data['contours'], poses):
            theta = math.atan2(s, c)
            cosine, sine = math.cos(theta), math.sin(theta)
            shapes.append(affine_transform(Polygon(contour), [cosine, -sine, sine, cosine, x, y]))
        left = min(p.bounds[0] for p in shapes)
        width = max(p.bounds[2] for p in shapes) - left
        widths.append(width)
        for i, shape in enumerate(shapes):
            vertices = [(x-left, y) for x, y in shape.exterior.coords]
            ax.add_patch(PatchPolygon(vertices, facecolor=plt.cm.tab20(i % 20), edgecolor='#496273', linewidth=.6, alpha=.7))
        ax.axhline(0, color='#445566', ls='--', lw=.8)
        ax.axhline(1200, color='#445566', ls='--', lw=.8)
        ax.set_ylim(-35, 1235)
        ax.set_aspect('equal')
        ax.set_title(f'{label}\nUtilization {metrics["strip_utilization"]:.2%} | Overlap sum {metrics["sum_pair_overlap_area"]:.6g}',
                     color='#16725d' if metrics['strict_valid'] else '#a34a35', fontsize=12)
        ax.set_xlabel('Horizontal span (display aligned at left)')
    for ax in axes:
        ax.set_xlim(-20, max(widths)+35)
    fig.suptitle(f'Sample {data["sample_index"]:03d}, candidate {index} | Official Dental weights', fontsize=15)
    fig.supxlabel('One development candidate; numerical reconstruction choices declared in docs/REPRODUCTION.md. Stored coordinates unchanged.', fontsize=9)
    fig.savefig(target, dpi=160)
    plt.close(fig)
    print(str(target.resolve()))


if __name__ == '__main__':
    main()
