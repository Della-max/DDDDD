"""Declared post-generation operator splitting of the paper's weighted SAT corrections.

Pair order and relaxation are numerical reconstruction choices: the paper does
not publish them. This is not round8's placement search, nor a claim of an exact
copy of the author's solver. Sampling still uses the simultaneous Eq. 9 field.
"""
import math

import numpy as np
from shapely.geometry import Polygon

from .geometry import CONTACT_MARGIN


def enhance_pairwise(geometry, actions, iterations=100, relaxation=1.75):
    if not 0 <= iterations <= 100 or not 0 < relaxation < 2 or not math.isfinite(relaxation):
        raise ValueError('Iterations must be in [0,100], relaxation in (0,2)')
    result = np.array(actions, dtype=np.float64, copy=True)
    oriented = geometry.oriented(result)
    shapes = oriented.shapes(result[:, :2])
    bounds = np.asarray([p.bounds for p in shapes])
    if np.any(bounds[:, 3] - bounds[:, 1] > geometry.height):
        return result, {'status': 'fixed_orientation_exceeds_height', 'iterations': 0,
                        'trace': [], 'metrics': geometry.measure(result)}
    trace, best = [], None
    best_iteration = None
    first_strict = None
    for iteration in range(iterations):
        previous = result[:, :2].copy()
        pair_updates = 0
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if bounds[i, 0] >= bounds[j, 2] or bounds[j, 0] >= bounds[i, 2] or bounds[i, 1] >= bounds[j, 3] or bounds[j, 1] >= bounds[i, 3]:
                    continue
                if shapes[i].intersection(shapes[j]).area == 0:
                    continue
                vector = relaxation * oriented.exit_vector(i, j, result[:, :2], shapes)
                total = geometry.areas[i] + geometry.areas[j]
                # Same per-pair area weights as Eq.9; updates are immediately visible.
                result[i, :2] += geometry.areas[j] / total * vector
                result[j, :2] -= geometry.areas[i] / total * vector
                for index in (i, j):
                    shapes[index] = Polygon(oriented.vertices[index] + result[index, :2])
                    bounds[index] = shapes[index].bounds
                pair_updates += 1
        for i in range(len(result)):
            delta = 0.
            if bounds[i, 1] < 0:
                delta = -bounds[i, 1] + CONTACT_MARGIN
            elif bounds[i, 3] > geometry.height:
                delta = geometry.height - bounds[i, 3] - CONTACT_MARGIN
            if delta:
                result[i, 1] += delta
                shapes[i] = Polygon(oriented.vertices[i] + result[i, :2])
                bounds[i] = shapes[i].bounds
        before_gap = geometry.measure(result)
        candidate, gap_moves = oriented.gap_sweep(result)
        measured = geometry.measure(candidate)
        gap_rejected = before_gap['strict_valid'] and not measured['strict_valid']
        if not gap_rejected:
            result = candidate
        else:
            measured = before_gap
        shapes = oriented.shapes(result[:, :2])
        bounds = np.asarray([p.bounds for p in shapes])
        movement = float(np.linalg.norm(result[:, :2] - previous, axis=1).max())
        trace.append({'iteration': iteration+1, 'pair_updates': pair_updates,
                      'relaxation': relaxation, 'max_movement': movement,
                      'gap_moves': gap_moves, 'gap_rejected_to_preserve_feasibility': gap_rejected,
                      **{k: v for k, v in measured.items() if k != 'overlap_pairs'}})
        if measured['strict_valid']:
            if first_strict is None:
                first_strict = iteration + 1
            if best is None or measured['span_x'] < best[1]['span_x']:
                best = result.copy(), measured
                best_iteration = iteration + 1
            if movement <= 1e-6:
                break
    if best is not None:
        result = best[0]
    return result, {'status': 'strict_valid' if best is not None else 'iteration_limit',
                    'iterations': len(trace), 'first_strict_iteration': first_strict,
                    'best_feasible_iteration': best_iteration, 'trace': trace,
                    'metrics': geometry.measure(result)}
