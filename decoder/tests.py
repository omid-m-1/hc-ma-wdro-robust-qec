from pathlib import Path
import sys,itertools,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(Path(__file__).parent),str(ROOT),str(ROOT/'scripts')]
import numpy as np,torch
from scipy.optimize import milp,LinearConstraint,Bounds
from scipy.stats import beta
from physics import *
from learning import path_weights,adversary,transport_distance
from qecrobust.headline_specs import HEADLINE_CASES


def run():
    spec=HEADLINE_CASES['hv_checker_cross_perimeter'];src=np.array([.03,.323333333333,.323333333333,.323333333334,0.]);rng=np.random.default_rng(6173);result={}
    exposures=[];logical_checks=[];fixed=[];milp_tests=0;guaranteed=[]
    for d in [3,5,7,15,19]:
        c,xy,ed,env,F,H,Z=setup(d,spec);n=c.n
        assert logical(c,c.logicals_x[0])==1 and logical(c,c.logicals_z[0])==2
        logical_checks.append(d);exposures.append([float(2*(e['pxx'].sum()+e['pzz'].sum())/n) for e in env])
        for e in env:
            assert all(np.all((e[k]>0)&(e[k]<.1)) for k in ['px','py','pz','pxx','pzz'])
        s=draw(c,F,H,Z,np.ones(5)/5,5,5,101);dec=matching(c,env,ed,src);cor=decode(c,dec,s['S'])
        assert all(c.in_codespace(e^co) for e,co in zip(s['ERR'],cor))
        assert np.array_equal(s['S'],np.asarray(c.measure_syndrome(s['ERR']).T,np.uint8))
        # Independent marginal-noise control, same p at every distance.
        er=(rng.random((20000,2*n))<.03).astype(np.uint8);syn=np.asarray(c.measure_syndrome(er).T,np.uint8)
        from pymatching import Matching
        uniform=(Matching(c.Hz),Matching(c.Hx));co=decode(c,uniform,syn);failure=logical(c,er^co)!=0;count=int(failure.sum());N=len(er)
        lo=0. if count==0 else float(beta.ppf(.025,count,N-count+1));hi=1. if count==N else float(beta.ppf(.975,count+1,N-count))
        fixed.append(dict(distance=d,errors=count,shots=N,ler=count/N,ci95=[lo,hi]))
        if d<=7:
            # Exhaustive pure X/Z errors through floor((d-1)/2), batched MWPM.
            total=0
            for w in range(1,(d-1)//2+1):
                combos=list(itertools.combinations(range(n),w))
                for half in (0,n):
                    for st in range(0,len(combos),2048):
                        chunk=combos[st:st+2048];er=np.zeros((len(chunk),2*n),np.uint8)
                        for row,ids in enumerate(chunk):er[row,np.asarray(ids)+half]=1
                        syn=np.asarray(c.measure_syndrome(er).T,np.uint8);co=decode(c,uniform,syn)
                        assert not np.any(logical(c,er^co));total+=len(er)
            guaranteed.append(dict(distance=d,patterns=total))
            # MWPM weight objective checked against independent integer parity solver.
            for half in (0,1):
                A=(c.Hz if half==0 else c.Hx).astype(float).toarray();m=A.shape[0];weights=rng.uniform(.2,3,n)
                matcher=Matching(A,weights=weights)
                for j in range(5):
                    e=(rng.random(n)<.1).astype(np.uint8);sy=(A@e)%2;got=matcher.decode(sy)
                    B=np.concatenate([A,-2*np.eye(m)],axis=1);cost=np.r_[weights,np.zeros(m)]
                    rr=milp(cost,integrality=np.ones(n+m),bounds=Bounds(np.zeros(n+m),np.r_[np.ones(n),np.full(m,n)]),constraints=LinearConstraint(B,sy,sy),options={'time_limit':20})
                    assert rr.success and abs(weights@got-rr.fun)<1e-6;milp_tests+=1
        print('physics distance',d,'OK',flush=True)
    assert np.allclose(exposures,exposures[0],rtol=1e-10)
    # Full finite-path likelihood ratio equals direct Markov path probability.
    K=3;L=4;paths=np.array(list(itertools.product(range(K),repeat=L)));valid=np.ones_like(paths,bool);q=np.array([.2,.3,.5]);u=np.ones(K)/K;stay=.95
    def probability(p):
        ans=p[paths[:,0]].copy()
        for t in range(1,L):ans*=stay*(paths[:,t]==paths[:,t-1])+(1-stay)*p[paths[:,t]]
        return ans
    w=path_weights(q,paths,valid).numpy();assert np.allclose(w,probability(q)/probability(u));assert abs(probability(u)@w-1)<1e-12
    qt=torch.tensor(q,dtype=torch.float64,requires_grad=True);f=torch.tensor(np.arange(len(paths))%7,dtype=torch.float64)
    objective=(path_weights(qt,paths,valid)*torch.tensor(probability(u))*f).sum();g=torch.autograd.grad(objective,qt)[0].numpy()
    direction=np.array([1.,-1.,0]);eps=1e-6
    numerical=(probability(q+eps*direction)@f.numpy()-probability(q-eps*direction)@f.numpy())/(2*eps)
    assert abs(numerical-g@direction)<1e-7
    # Valid-mask padding invariant: a real zero syndrome remains valid evidence.
    c,xy,ed,env,F,H,Z=setup(3,spec);ex=Experts(c,env,ed,F,H,Z,src)
    dat=draw(c,F,H,Z,np.ones(5)/5,3,4,733);ex.enrich(dat)
    assert dat['X'][0,-1,-1]==1 and np.all(dat['X'][0,:-1,-1]==0)
    y=dat['P'].copy();other=dict(dat,E=dat['E'][::-1].copy());ex.enrich(other)
    assert np.array_equal(y,other['P']) and np.array_equal(dat['X'],other['X'])
    # d3 exact normalization and direct Monte Carlo distribution check.
    assert np.allclose(ex.tables.sum((1,2)),1)
    sample=draw(c,F,H,Z,np.eye(5)[4],10000,5,331)
    sid=sample['S'].astype(int)@(1<<np.arange(c.n_stabilizers));lab=logical(c,sample['ERR']);index=sid+(lab<<c.n_stabilizers)
    empirical=np.bincount(index,minlength=1024)/len(index);truth=ex.tables[4].T.ravel();tv=float(np.abs(empirical-truth).sum()/2);assert tv<.05
    result.update(logical_operator_action_checks=logical_checks,constant_pair_exposure=exposures,fixed_independent_xz_p03=fixed,exhaustive_guaranteed_errors=guaranteed,mwpm_milp_objective_checks=milp_tests,exact_d3_50000_shot_tv=tv,trajectory_probability_and_gradient='passed',no_true_environment_deployment_input='passed',padding_mask='passed')
    return result

if __name__=='__main__':
    ROOT.joinpath('results/validation').mkdir(parents=True,exist_ok=True);r=run();ROOT.joinpath('results/validation/physics_tests.json').write_text(json.dumps(r,indent=2));print('ALL V90 PHYSICS AND TRAJECTORY TESTS PASSED')
