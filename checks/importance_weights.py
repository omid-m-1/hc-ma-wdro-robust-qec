"""SNIS definition, differentiability and unchanged raw estimator."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'decoder'),str(ROOT),str(ROOT/'scripts')]
import numpy as np,torch
from learning import path_weights,history_weights
path=np.array([[0,0,1],[1,1,1],[2,0,0],[2,2,2]]);valid=np.ones_like(path,bool)
q=torch.tensor([.2,.3,.5],dtype=torch.float64,requires_grad=True)
a=path_weights(q,path,valid);b=history_weights(q,path,valid,normalization='self_normalized')
assert torch.equal(a,history_weights(q,path,valid)) and torch.allclose(b,a/a.mean()) and torch.allclose(b.mean(),torch.ones((),dtype=b.dtype))
loss=(b*torch.tensor([1.,0.,.5,.25])).mean();grad=torch.autograd.grad(loss,q)[0];eps=1e-6
for k in range(3):
 qq=q.detach().clone();qq[k]+=eps;plus=(history_weights(qq,path,valid,normalization='self_normalized')*torch.tensor([1.,0.,.5,.25])).mean()
 qq[k]-=2*eps;minus=(history_weights(qq,path,valid,normalization='self_normalized')*torch.tensor([1.,0.,.5,.25])).mean()
 assert abs(float((plus-minus)/(2*eps)-grad[k]))<1e-6
print('Raw importance ratios unchanged; SNIS unit mean and objective gradients passed. SNIS is finite-sample biased, not an unbiased exact integral.')
