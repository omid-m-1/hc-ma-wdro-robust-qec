from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(Path(__file__).parent),str(ROOT),str(ROOT/'scripts')]
from physics import *
from learning import *
from qecrobust.headline_specs import HEADLINE_CASES
c,xy,ed,env,F,H,Z=setup(5,HEADLINE_CASES['hv_checker_cross_perimeter']);src=np.array([.03,.323333333333,.323333333333,.323333333334,0.]);ex=Experts(c,env,ed,F,H,Z,src,expert_calibration='action')
cal=draw(c,F,H,Z,np.ones(5)/5,100,5,7123);ex.enrich(cal,calibrate=True);assert np.allclose(ex.action_conf.sum(2),1)
data=draw(c,F,H,Z,np.ones(5)/5,4,5,6138);ex.enrich(data);P=data['P'].copy();X=data['X'].copy();data['E']=data['E'][::-1].copy();ex.enrich(data);assert np.array_equal(P,data['P']) and np.array_equal(X,data['X'])
q=adversary(np.arange(len(data['Y']))%3,data['path'],data['valid'],src,np.ones((5,5))-np.eye(5),.5)
assert abs(q.sum()-1)<1e-7 and np.min(q)>-1e-9 and transport_distance(q,src,np.ones((5,5))-np.eye(5))<=.5+1e-7
r1=np.array([.2,.2,.2,.2,.2]);r2=np.array([.4,.1,.1,.1,.3]);a=(path_weights(r1,data['path'],data['valid'])+path_weights(r2,data['path'],data['valid']))/2;b=path_weights((r1+r2)/2,data['path'],data['valid']);assert not torch.allclose(a,b)
print('ACTION-CONDITIONAL NORMALIZATION, NO TRUE-LABEL INPUT, TRANSPORT FEASIBILITY, AND PATH-REPLAY MIXTURE CHECKS PASSED')
