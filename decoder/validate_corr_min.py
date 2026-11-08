from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(Path(__file__).parent),str(ROOT),str(ROOT/'scripts')]
from physics import *
from qecrobust.headline_specs import HEADLINE_CASES,HEADLINE_SOURCE_PI
from scipy.optimize import milp,Bounds,LinearConstraint
c,xy,ed,env,F,H,Z=setup(3,HEADLINE_CASES['hv_checker_cross_perimeter']);z=np.asarray(HEADLINE_SOURCE_PI)@Z;M=len(z)
d=draw(c,F,H,Z,np.ones(5)/5,8,1,877);pred=component_minweight_prediction(c,F,H,z,d['S']);count=0
for bit,rows in [(0,c.z_indices),(1,c.x_indices)]:
 rows=np.flatnonzero(rows) if np.asarray(rows).dtype==bool else np.asarray(rows);part=F.toarray().T.copy()
 if bit==0:part[:,c.n:]=0
 else:part[:,:c.n]=0
 A=np.vstack([H[rows].toarray(),(logical(c,part)>>bit)&1]);m=len(A);matrix=np.c_[A,-2*np.eye(m)];cost=np.r_[np.log((1-z)/z),np.zeros(m)]
 for syn,pr in zip(d['S'],pred):
  vals=[]
  for label in [0,1]:
   rhs=np.r_[syn[rows],label];r=milp(cost,integrality=np.ones(M+m),bounds=Bounds(np.zeros(M+m),np.r_[np.ones(M),np.full(m,M)]),constraints=LinearConstraint(matrix,rhs,rhs),options={'time_limit':20});assert r.success;vals.append(r.fun);count+=1
  chosen=(pr>>bit)&1;assert vals[chosen]<=min(vals)+1e-6
print('Exact CSS Corr-Min agrees with',count,'independent constrained MILP objectives')
