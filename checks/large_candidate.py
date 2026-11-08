"""New candidate: causal expert features and mixture-envelope reference."""
from pathlib import Path
import sys,copy
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'decoder'),str(ROOT),str(ROOT/'scripts')]
import numpy as np
from learning import action_history
rng=np.random.default_rng(117);P=rng.random((8,5,4));P/=P.sum(2,keepdims=True)
d=dict(P=P,X=np.zeros((8,2,3)),Y=np.zeros(8,int),E=np.zeros(8,int),nseq=2,rounds=4)
a=action_history(copy.deepcopy(d))['X'];q=copy.deepcopy(d);q['Y'][:]=3;q['E'][:]=4
assert np.array_equal(a,action_history(q)['X'])
q=copy.deepcopy(d);q['P'][3]=.25;b=action_history(q)['X'];assert np.array_equal(a[:3],b[:3])
assert np.all(a[[0,4],0,3:]==0)
g=rng.random((8,5));g/=g.sum(1,keepdims=True);y=rng.integers(0,4,8)
mixture=np.einsum('nk,nkl->nl',g,P);reference=P[np.arange(8),:,y].max(1)
assert np.all(np.log(reference/mixture[np.arange(8),y])>=-1e-12)
assert np.isclose(a.shape[-1],23)
print('Causal action histories, label/environment invariance, padding and nonnegative mixture-NLL envelope passed.')
