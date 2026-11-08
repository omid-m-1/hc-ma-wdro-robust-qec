"""Verify explicit distance policy, extra-case counts and source equivalence."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'decoder'),str(ROOT),str(ROOT/'scripts')]
import numpy as np
from physics import setup
from qecrobust.case_bank import get_case,names
from qecrobust.experiments import PRESETS
spec=get_case('band_family_topology')['specs']
for d in [3,5,7,15,19]:
    resolved='segments' if d==5 else 'long_segments'
    a=setup(d,spec,noise_geometry='distance_scaled')
    b=setup(d,spec,noise_geometry=resolved)
    assert (a[4]!=b[4]).nnz==0 and (a[5]!=b[5]).nnz==0
    assert np.array_equal(a[6],b[6])
    assert all(e['geometry_variant']==resolved for e in a[3])
    print('distance policy',d,resolved,'L',a[3][0]['burst_length'],'matches explicit setup')
assert len(names('hv'))==4 and len(names('persistent'))==7
assert len(names('hv_band'))==5 and len(set(names('hv_band')))==5
assert len(names('all'))==11 and names('band')==['band_family_topology']
for key in PRESETS['v98-long-average']:
    if key!='noise_geometry':assert PRESETS['v10-main'][key]==PRESETS['v98-long-average'][key]
print('v10 distance policy and case grids passed; training settings unchanged.')
