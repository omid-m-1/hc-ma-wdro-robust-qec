"""Low-rank implementation of mechanism-aware logical-competition metrics.

The witness is sampled from competing syndrome-consistent fault candidates,
not from an exhaustive logical-coset partition function.
"""
import numpy as np
from scipy import sparse
from panqec_distance_backend import _metric_closure,_robust_q
from structured_mechanisms import logical

def base_precision(code,F,H):
 n=code.n;M=F.shape[1]
 B=H.astype(float);bn=np.sqrt(np.asarray(B.power(2).sum(0)).ravel());W=(B.T@B).tocoo();W.data/=np.maximum(bn[W.row]*bn[W.col],1e-12)
 L=np.asarray([[(logical(code,F[:,j].toarray().ravel())>>b)&1 for j in range(M)] for b in range(2)]).T
 W.data/=1+np.abs(L[W.row]-L[W.col]).sum(1);W=W.tocsr();W.setdiag(0);W.eliminate_zeros()
 P=(F[:n]+F[n:]).astype(float);P.data[:]=1;pn=np.sqrt(np.asarray(P.power(2).sum(0)).ravel());P=P@sparse.diags(1/np.maximum(pn,1));Wp=(P.T@P).tocsr();Wp.setdiag(0);Wp.eliminate_zeros()
 def lapnorm(w):
  l=sparse.diags(np.asarray(w.sum(1)).ravel())-w;t=l.diagonal().mean()
  return l/max(t,1e-12)
 Lb=.5*lapnorm(W)+.5*lapnorm(Wp);return (sparse.eye(M)+Lb)/2

def witness(p,faults,classes,mode):
 # Best fault candidate per logical coset; top two cosets compete at this p.
 p=np.clip(p,1e-6,.49);ll=np.log(p/(1-p));hh=[];ww=[]
 for ff,cc in zip(faults,classes):
  scores=ff@ll;best=[]
  for c in np.unique(cc):
   ids=np.flatnonzero(cc==c);j=ids[np.argmax(scores[ids])];best.append(j)
  if len(best)<2:continue
  ids=sorted(best,key=lambda j:scores[j],reverse=True);a,b=ids[:2]
  h=(ff[a].astype(float)-ff[b])/(1-p)
  if np.linalg.norm(h)<1e-12:continue
  hh.append(h);ww.append(1/(1+np.exp(min(abs(scores[a]-scores[b]),60))))
 if not hh:return np.zeros((0,len(p)))
 h=np.asarray(hh);w=np.asarray(ww);w/=w.sum()
 if mode=='paper':h/=np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-12)
 U=np.sqrt(w)[:,None]*h
 if mode=='legacy':U/=max(np.linalg.norm(U),1e-12)
 return U

def metric_costs(Z,src,A,faults,classes):
 M=Z.shape[1];p0=src@Z
 def make(p,mode):
  U=witness(p,faults,classes,mode);AU=np.asarray(A@U.T)
  inv=np.linalg.inv(np.eye(len(U))+U@AU) if len(U) else np.zeros((0,0))
  tr=float(A.diagonal().sum())-(np.trace(inv@(AU.T@AU)) if len(U) else 0.)
  def quadratic(delta):
   delta=delta/np.maximum(p,.001);v=AU.T@delta
   return max(0.,(delta@(A@delta)-v@inv@v)*M/max(tr,1e-12))
  return quadratic,len(U)
 def norm(C):
  v=C[C>1e-12];return C/np.median(v) if len(v) else C
 def cost(fn):
  C=np.zeros((len(Z),len(Z)))
  for i in range(len(Z)):
   for j in range(i+1,len(Z)):C[i,j]=C[j,i]=np.sqrt(fn(Z[j]-Z[i]))
  return norm(C)
 fn,nw=make(p0,'paper');Cp=cost(fn);fl,_=make(p0,'legacy');Cl=cost(fl)
 fns=[make(z,'paper')[0] for z in Z];Ce=np.zeros_like(Cp)
 for i in range(len(Z)):
  for j in range(i+1,len(Z)):
   delta=Z[j]-Z[i];Ce[i,j]=Ce[j,i]=np.sqrt(.5*(fns[i](delta)+fns[j](delta)))
 Ce=norm(Ce);Cm=norm(_metric_closure(np.minimum(Cp,Ce)))
 D=cost(lambda d:np.sum((d/np.maximum(p0,.001))**2))
 return {'euclid':D,'paper_norm':Cp,'legacy_regret_rawmag':Cl,'local_ma_regret':Ce,'multiscale_ma_regret':Cm},nw

def mixtures(risks,reg,src,costs,rho=.5,kappa=3):
 K=len(src);h=(reg-reg.min())/np.ptp(reg) if np.ptp(reg)>1e-12 else np.zeros(K)
 C=costs['multiscale_ma_regret'];Ch=_metric_closure(C/(1+kappa*.5*(h[:,None]+h[None,:])))
 costs=dict(costs,hc_multiscale_ma_regret=Ch,tail_wdro=np.ones((K,K))-np.eye(K))
 out={'erm':src.copy()}
 for m,C in costs.items():out[m]=_robust_q(reg if m in ['legacy_regret_rawmag','local_ma_regret','multiscale_ma_regret','hc_multiscale_ma_regret'] else risks,src,C,rho)
 q=src.copy();pos=np.flatnonzero(src>0);r=pos[np.argmin(src[pos])];delta=min(rho,1-src[r]);others=pos[pos!=r];q[others]*=(q[others].sum()-delta)/q[others].sum();q[r]+=delta;out['rare_tail_wdro']=q
 return out,costs,h
