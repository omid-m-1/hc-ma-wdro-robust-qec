"""Averaging preserves probability semantics and uses no labels/environment."""
from pathlib import Path
import sys,copy
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'decoder'),str(ROOT),str(ROOT/'scripts')]
import numpy as np,torch
from learning import RegimeGRU,PredictiveAverage,prediction
rng=np.random.default_rng(119);torch.manual_seed(119)
mods=[RegimeGRU(7,5) for _ in range(3)];model=PredictiveAverage(mods)
P=rng.random((10,5,4));P/=P.sum(2,keepdims=True)
d=dict(X=rng.random((10,8,7)).astype(np.float32),P=P,Y=np.zeros(10,int),E=np.zeros(10,int))
p,g=prediction(model,d);individual=[prediction(m,d) for m in mods]
assert np.allclose(p,np.mean([v[0] for v in individual],0)) and np.allclose(g,np.mean([v[1] for v in individual],0))
d['Y'][:]=3;d['E'][:]=4;assert np.array_equal(p,prediction(model,d)[0]);assert np.allclose(p.sum(1),1)
print('Predictive averaging equals averaged probabilities; normalization and label/environment invariance passed.')
