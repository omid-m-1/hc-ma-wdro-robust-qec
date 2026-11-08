"""Trajectory-correct, replay-stabilized offline Markov DRO (non-GNN)."""
import copy
import numpy as np
import torch
from torch import nn
from scipy.optimize import linprog
METHODS=['erm','hc','tail','rare_tail','euclid','paper_ma','legacy_ma','local_ma','multiscale_ma']


def path_weights(q,path,valid,stay=.95):
    """Exact stationary finite-window likelihood ratio to uniform proposal.

    Products, rather than log(q), preserve derivatives on simplex boundaries.
    Includes the first valid state's prior and every transition in the window.
    """
    if not torch.is_tensor(q):q=torch.tensor(q,dtype=torch.float64)
    path=torch.as_tensor(path,dtype=torch.long,device=q.device);valid=torch.as_tensor(valid,device=q.device)
    K=len(q);w=torch.ones(len(path),dtype=q.dtype,device=q.device);started=torch.zeros(len(path),dtype=torch.bool,device=q.device)
    for t in range(path.shape[1]):
        v=valid[:,t];first=v&~started;later=v&started
        firstfactor=K*q[path[:,t]]
        if t:
            same=(path[:,t]==path[:,t-1]).to(q.dtype)
            factor=(stay*same+(1-stay)*q[path[:,t]])/(stay*same+(1-stay)/K)
        else:factor=torch.ones_like(w)
        w=w*torch.where(first,firstfactor,torch.where(later,factor,torch.ones_like(w)));started|=v
    return w


def history_weights(q,path,valid,stay=.95,normalization='none'):
    w=path_weights(q,path,valid,stay)
    if normalization=='self_normalized':w=w/w.mean().clamp_min(1e-12)
    elif normalization!='none':raise ValueError(normalization)
    return w


def transport_oracle(score,src,C,rho):
    K=len(src);obj=-np.tile(score,K);A=np.zeros((K,K*K))
    for i in range(K):A[i,i*K:(i+1)*K]=1
    r=linprog(obj,A_ub=C.reshape(1,-1),b_ub=[rho],A_eq=A,b_eq=src,bounds=(0,None),method='highs')
    if not r.success:raise RuntimeError(r.message)
    T=r.x.reshape(K,K);assert np.allclose(T.sum(1),src) and (T*C).sum()<=rho+1e-7
    return T.sum(0)


def transport_distance(q,src,C):
    K=len(src);A=[];b=[]
    for i in range(K):a=np.zeros((K,K));a[i]=1;A.append(a.ravel());b.append(src[i])
    for j in range(K):a=np.zeros((K,K));a[:,j]=1;A.append(a.ravel());b.append(q[j])
    r=linprog(C.ravel(),A_eq=A,b_eq=b,bounds=(0,None),method='highs')
    if not r.success:raise RuntimeError(r.message)
    return float(r.fun)


def feasible_rays(src,C,rho):
    points=[np.asarray(src,float).copy()]
    for k in range(len(src)):
        vertex=np.eye(len(src))[k];cost=float(np.asarray(src)@C[:,k]);a=min(1.,rho/max(cost,1e-12))
        points.append((1-a)*np.asarray(src)+a*vertex)
    return points

def adversary(loss,path,valid,src,C,rho,start=None,steps=4,search='local',stay=.95,normalization='none'):
    q=np.asarray(src if start is None else start,float).copy();loss=torch.tensor(loss,dtype=torch.float64)
    if search=='rays':
        points=[q]+feasible_rays(src,C,rho)
        values=[float((history_weights(p,path,valid,stay,normalization)*loss).mean()) for p in points]
        q=points[int(np.argmax(values))].copy()
    for t in range(steps):
        qt=torch.tensor(q,dtype=torch.float64,requires_grad=True);val=(history_weights(qt,path,valid,stay,normalization)*loss).mean()
        grad=torch.autograd.grad(val,qt)[0].numpy();candidate=transport_oracle(grad,src,C,rho)
        # Feasible Frank-Wolfe step with a deterministic finite line search.
        choices=[q]+[(1-a)*q+a*candidate for a in (.25,.5,.75,1.)]
        values=[float((history_weights(p,path,valid,stay,normalization)*loss).mean()) for p in choices];q=choices[int(np.argmax(values))]
    assert transport_distance(q,src,C)<=rho+1e-6
    return q


class RegimeGRU(nn.Module):
    def __init__(self,nin,K):
        super().__init__();self.gru=nn.GRU(nin,48,batch_first=True);self.out=nn.Linear(48,K)
    def forward(self,x):return self.out(self.gru(x)[0][:,-1]).softmax(1)


class PredictiveAverage(nn.Module):
    def __init__(self,models):
        super().__init__();self.models=nn.ModuleList(models)
    def forward(self,x):return torch.stack([model(x) for model in self.models]).mean(0)


def prediction(model,data):
    model.eval();out=[];gp=[]
    with torch.no_grad():
        for st in range(0,len(data['Y']),1024):
            g=model(torch.tensor(data['X'][st:st+1024]));p=torch.einsum('nk,nkl->nl',g,torch.tensor(data['P'][st:st+1024],dtype=torch.float32));out.append(p.numpy());gp.append(g.numpy())
    return np.concatenate(out),np.concatenate(gp)


def fit(method,train,val,src,costs,floor,seed=47,epochs=14,rho=.5,optimizer_mode='iterative',adversary_loss='nll',adversary_search='local',train_stay=.95,regret_reference='environment',auxiliary_target='environment',weight_normalization='none',checkpoint_mode='last'):
    torch.manual_seed(seed);model=RegimeGRU(train['X'].shape[-1],len(src));opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
    C=costs.get(method,np.ones((len(src),len(src)))-np.eye(len(src)))
    q=src.copy();replay=[];trace=[];best=None;bestscore=float('inf');snapshots=[]
    X=torch.tensor(train['X']);P=torch.tensor(train['P'],dtype=torch.float32);Y=torch.tensor(train['Y']);E=torch.tensor(train['E']);allids=torch.arange(len(Y))
    # Uniform Markov proposal with full support; exact finite-history weighting.
    def weights(pi,data):return history_weights(pi,data['path'],data['valid'],train_stay,weight_normalization).numpy()
    for ep in range(epochs):
        vp,_=prediction(model,val);loss=-np.log(np.maximum(vp[np.arange(len(vp)),val['Y']],1e-10))
        # HC and regret variants subtract an independent candidate-conditioned
        # reference NLL, preserving per-example trajectory dependence.
        if adversary_loss=='logical_error':
            loss=(vp.argmax(1)!=val['Y']).astype(float)
            reference=(val['P'][np.arange(len(vp)),val['E']].argmax(1)!=val['Y']).astype(float)
        else:reference=floor
        if regret_reference=='selector':reference=(val['P'].argmax(2)!=val['Y'][:,None]).all(1).astype(float)
        elif regret_reference=='soft_selector':reference=-np.log(np.maximum(val['P'][np.arange(len(vp)),:,val['Y']].max(1),1e-10))
        score=loss-reference if method in ['hc','legacy_ma','local_ma','multiscale_ma'] else loss
        if method=='erm':q=src.copy()
        elif method=='rare_tail':
            q=src.copy();k=np.flatnonzero(src>0)[np.argmin(src[src>0])];a=min(rho,1-src[k]);others=np.arange(len(src))!=k;q[others]*=(1-src[k]-a)/(1-src[k]);q[k]+=a
        elif ep==0 or optimizer_mode=='iterative':q=adversary(score,val['path'],val['valid'],src,C,rho,start=q,search=adversary_search,stay=train_stay,normalization=weight_normalization)
        # Audit feasible rays for BOTH optimizers, without using any test samples.
        raypts=feasible_rays(src,C,rho)
        rayvalues=[float((history_weights(p,val['path'],val['valid'],train_stay,weight_normalization)*torch.tensor(score,dtype=torch.float64)).mean()) for p in raypts]
        qvalue=float((history_weights(q,val['path'],val['valid'],train_stay,weight_normalization)*torch.tensor(score,dtype=torch.float64)).mean())
        missed_ray=max(rayvalues)-qvalue
        replay.append(q.copy())
        # Mixture of complete path distributions; weighting by the mean q
        # would be incorrect because Markov trajectory probabilities are nonlinear.
        active=replay if optimizer_mode=='iterative' else replay[:1]
        w=np.mean([weights(p,train) for p in active],axis=0);wt=torch.tensor(w,dtype=torch.float32)
        order=np.random.default_rng(seed+ep).permutation(len(Y));model.train();acc=0
        for st in range(0,len(order),256):
            z=torch.tensor(order[st:st+256]);g=model(X[z]);p=torch.einsum('nk,nkl->nl',g,P[z]);nll=-torch.log(p[torch.arange(len(z)),Y[z]].clamp_min(1e-8))
            envloss=-torch.log(g[torch.arange(len(z)),E[z]].clamp_min(1e-8))
            if auxiliary_target=='correct_expert':
                good=(P[z].argmax(2)==Y[z,None]).to(g.dtype)
                envloss=torch.where(good.any(1),-torch.log((g*good).sum(1).clamp_min(1e-8)),torch.zeros(len(z)))
            objective=((nll+.2*envloss)*wt[z]).mean();opt.zero_grad();objective.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();acc+=float(objective.detach())
        vp,_=prediction(model,val);error=(vp.argmax(1)!=val['Y']).astype(float)
        vw=np.mean([weights(p,val) for p in active],axis=0)
        # Validation uses a growing replay objective; don't compare scores
        # across different objectives to restore a stale early epoch.
        trace.append(dict(adversary_search=adversary_search,adversary_objective=qvalue,best_ray_objective=max(rayvalues),missed_ray_gain=max(0.,missed_ray),best_ray_candidate=int(np.argmax(rayvalues))-1,deployment_ray_objective=rayvalues[-1],epoch=ep+1,q=q.tolist(),replay_size=len(active),proposal_ess=float(w.sum()**2/max(w@w,1e-12)),weight_mean=float(w.mean()),train_objective=acc/max(1,len(order)/256),validation_replay_ler=float(np.average(error,weights=vw)),transport_cost=0. if method in ['erm','rare_tail'] else transport_distance(q,src,C)))
        if checkpoint_mode=='late_average' and ep>=epochs//2:snapshots.append(copy.deepcopy(model).eval())
    if checkpoint_mode=='late_average':model=PredictiveAverage(snapshots)
    elif checkpoint_mode!='last':raise ValueError(checkpoint_mode)
    if trace:trace[-1].update(checkpoint_mode=checkpoint_mode,predictive_members=len(snapshots) if snapshots else 1)
    # Fixed epoch budget, no test-dependent early stopping.
    return model,trace


def action_history(data):
    """Causal calibrated expert probabilities; never uses E/Y as features."""
    if data.get('action_history_added'):return data
    from physics import history_indices
    idx,valid=history_indices(data['nseq'],data['rounds'],data['X'].shape[1])
    extra=data['P'][idx].reshape(*idx.shape,-1).astype(np.float32);extra[~valid]=0
    data['X']=np.concatenate([data['X'],extra],axis=2);data['action_history_added']=True
    return data
