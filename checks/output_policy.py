"""Late averaging does not alter training; ray search covers audited candidates."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'decoder'),str(ROOT),str(ROOT/'scripts')]
import numpy as np,torch
from learning import fit,prediction
rng=np.random.default_rng(716);K=3;N=40
P=rng.random((N,K,4));P/=P.sum(2,keepdims=True);E=rng.integers(K,size=N);Y=rng.integers(4,size=N)
d=dict(X=rng.random((N,1,2)).astype(np.float32),P=P,E=E,Y=Y,path=E[:,None],valid=np.ones((N,1),bool))
src=np.ones(K)/K;C={'hc':np.ones((K,K))-np.eye(K)};floor=-np.log(P[np.arange(N),E,Y])
a,ta=fit('hc',d,d,src,C,floor,seed=137,epochs=4,adversary_loss='logical_error',adversary_search='rays')
b,tb=fit('hc',d,d,src,C,floor,seed=137,epochs=4,adversary_loss='logical_error',adversary_search='rays',checkpoint_mode='late_average')
assert len(b.models)==2 and all(torch.equal(v,b.models[-1].state_dict()[k]) for k,v in a.state_dict().items())
assert np.array_equal(prediction(a,d)[0],prediction(b.models[-1],d)[0])
assert max(t['missed_ray_gain'] for t in ta)<1e-8
print('Late averaging preserves training/last-iterate parameters; corrected adversary covers all audited rays.')
