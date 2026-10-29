"""Read verified official resources and construct the unchanged model's graph inputs."""
import math
import pickle

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from shapely.geometry import Polygon

from src.geometry_utils import Polygon as OfficialPolygon
from .assets import ROOT, TEST_SHA256, sha256


def load_resources(sample_index):
    root = ROOT / 'data' / 'official_dental'
    path = root / 'dental_test.pkl'
    if sha256(path) != TEST_SHA256:
        raise ValueError('Test set is not the verified official Dental test PKL')
    # Only deserialize the fixed, hash-verified author's artifact.
    with path.open('rb') as stream:
        ids = pickle.load(stream)
        poses = pickle.load(stream)
    if not 0 <= sample_index < len(ids):
        raise IndexError(sample_index)
    contours = [OfficialPolygon(root / 'polygons' / f'{i}.txt').getMaxContour()
                for i in range(440)]
    selected_ids = np.asarray(ids[sample_index], dtype=np.int64).reshape(-1)
    if selected_ids.shape != (48,) or selected_ids.min() < 0 or selected_ids.max() >= 440:
        raise ValueError('Expected 48 Dental pieces with valid polygon IDs')
    return contours, selected_ids, np.asarray(poses[sample_index], dtype=np.float64).reshape(-1, 4)


def make_features(contours):
    graphs = []
    for contour in contours:
        p = np.asarray(contour)
        previous = np.roll(p, 1, axis=0) - p
        following = np.roll(p, -1, axis=0) - p
        dot = (previous * following).sum(axis=1)
        cross = previous[:, 0] * following[:, 1] - previous[:, 1] * following[:, 0]
        angle = np.abs(np.arctan2(cross, dot)) * 180 / math.pi
        angle = np.where(cross < 0, 360 - angle, angle)
        nodes = torch.tensor(np.column_stack([p, angle]), dtype=torch.float32)
        n = len(p)
        edges = [(i, (i+1) % n) for i in range(n)] + [((i+1) % n, i) for i in range(n)]
        shape = Polygon(p)
        graphs.append(Data(x=nodes, edge_index=torch.tensor(edges).T.contiguous(),
                           area=torch.tensor(shape.area, dtype=torch.float32),
                           perm=torch.tensor(shape.length, dtype=torch.float32)))
    return Batch.from_data_list(graphs)
