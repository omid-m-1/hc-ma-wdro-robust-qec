from __future__ import annotations

# QECROBUST_PORTABLE_BOOTSTRAP
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# END QECROBUST_PORTABLE_BOOTSTRAP

import numpy as np

GRID = np.arange(9).reshape(3,3)
EDGES_H = [(0,1),(1,2),(3,4),(4,5),(6,7),(7,8)]
EDGES_V = [(0,3),(3,6),(1,4),(4,7),(2,5),(5,8)]
EDGES = EDGES_H + EDGES_V


def _field(default=0.85):
    return np.full(9, float(default), dtype=float)

def local_pattern(name: str) -> np.ndarray:
    """Spatial rate multipliers on the 3x3 data-qubit grid."""
    if name in {'uniform','none'}:
        return np.ones(9)
    a = _field(0.85)
    masks = {
        'hotspot_left_col':[0,3,6], 'hotspot_mid_col':[1,4,7], 'hotspot_right_col':[2,5,8],
        'hotspot_top_row':[0,1,2], 'hotspot_mid_row':[3,4,5], 'hotspot_bottom_row':[6,7,8],
        'hotspot_tl_2x2':[0,1,3,4], 'hotspot_tr_2x2':[1,2,4,5],
        'hotspot_bl_2x2':[3,4,6,7], 'hotspot_br_2x2':[4,5,7,8],
        'l_tl':[0,1,3], 'l_tr':[1,2,5], 'l_bl':[3,6,7], 'l_br':[5,7,8],
        'checker_even':[0,2,4,6,8], 'checker_odd':[1,3,5,7],
        'diag_main':[0,4,8], 'diag_anti':[2,4,6],
        'cross':[1,3,4,5,7],
        'band_h_top':[0,1,2,3,4,5], 'band_h_bottom':[3,4,5,6,7,8],
        'band_v_left':[0,1,3,4,6,7], 'band_v_right':[1,2,4,5,7,8],
    }
    if name in masks:
        a[masks[name]] = 1.45
        return a
    if name == 'gradient_lr':
        return np.tile(np.array([0.80,1.10,1.45]),3)
    if name == 'gradient_rl':
        return np.tile(np.array([1.45,1.10,0.80]),3)
    if name == 'gradient_tb':
        return np.repeat(np.array([0.80,1.10,1.45]),3)
    if name == 'gradient_bt':
        return np.repeat(np.array([1.45,1.10,0.80]),3)
    raise ValueError(f'unknown local pattern {name}')


def pair_pattern(name: str) -> np.ndarray:
    """Multipliers over all 12 nearest-neighbor edges in EDGES order."""
    m = np.full(len(EDGES), 0.35, dtype=float)
    if name in {'uniform','all'}:
        return np.ones(len(EDGES))
    selected = []
    if name == 'horizontal_all': selected = EDGES_H
    elif name == 'vertical_all': selected = EDGES_V
    elif name == 'horizontal_row0': selected = [(0,1),(1,2)]
    elif name == 'horizontal_row1': selected = [(3,4),(4,5)]
    elif name == 'horizontal_row2': selected = [(6,7),(7,8)]
    elif name == 'vertical_col0': selected = [(0,3),(3,6)]
    elif name == 'vertical_col1': selected = [(1,4),(4,7)]
    elif name == 'vertical_col2': selected = [(2,5),(5,8)]
    elif name == 'mixed_a': selected = [(0,1),(3,4),(6,7),(2,5),(5,8)]
    elif name == 'mixed_b': selected = [(1,2),(4,5),(7,8),(0,3),(3,6)]
    elif name == 'center_cross': selected = [(1,4),(4,7),(3,4),(4,5)]
    elif name == 'perimeter': selected = [(0,1),(1,2),(2,5),(5,8),(7,8),(6,7),(3,6),(0,3)]
    else: raise ValueError(f'unknown pair pattern {name}')
    idx={e:i for i,e in enumerate(EDGES)}
    for e in selected:
        e=tuple(sorted(e)); m[idx[e]]=1.45
    return m

LOCAL_PATTERNS = [
    'hotspot_left_col','hotspot_mid_col','hotspot_right_col',
    'hotspot_top_row','hotspot_mid_row','hotspot_bottom_row',
    'hotspot_tl_2x2','hotspot_tr_2x2','hotspot_bl_2x2','hotspot_br_2x2',
    'gradient_lr','gradient_rl','gradient_tb','gradient_bt',
    'l_tl','l_tr','l_bl','l_br','checker_even','checker_odd',
    'band_h_top','band_h_bottom','band_v_left','band_v_right','cross','uniform'
]
PAIR_PATTERNS = [
    'horizontal_all','vertical_all','horizontal_row0','horizontal_row1','horizontal_row2',
    'vertical_col0','vertical_col1','vertical_col2','mixed_a','mixed_b','center_cross','perimeter','uniform'
]
