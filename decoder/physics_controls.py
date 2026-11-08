from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(Path(__file__).parent),str(ROOT),str(ROOT/'scripts')]
from physics import *
ROOT.joinpath("results/validation").mkdir(parents=True,exist_ok=True)
from scipy.stats import beta
from qecrobust.headline_specs import HEADLINE_CASES,HEADLINE_SOURCE_PI,HEADLINE_TEST_PI

def run(shots=20000):
 rows=[]
 for case,spec in HEADLINE_CASES.items():
  if not case.startswith('hv_'):continue
  for d in [3,5,7,15,19]:
   for profile in ['audit_original','strength_only','exposure_only','matched']:
    c,xy,ed,env,F,H,Z=setup(d,spec,'original')
    for e in env:
     if profile in ['strength_only','matched']:
      for k in ['px','py','pz','pxx','pzz']:e[k]*=.6
     if profile in ['exposure_only','matched']:
      ratio=(12/9)/(len(ed)/c.n)
      for k in ['pxx','pzz']:e[k]*=ratio
    Z=np.stack([np.r_[e['px'],e['py'],e['pz'],e['pxx'],e['pzz']] for e in env])
    dec=matching(c,env,ed,HEADLINE_SOURCE_PI)
    # Same held-out random stream across physical controls; no learner involved.
    data=draw(c,F,H,Z,np.asarray(HEADLINE_TEST_PI),shots,1,820000+d,.98)
    corr=decode(c,dec,data['S']);fail=logical(c,data['ERR']^corr)!=0;errors=int(fail.sum());N=len(fail)
    lo=0 if not errors else float(beta.ppf(.025,errors,N-errors+1));hi=1 if errors==N else float(beta.ppf(.975,errors+1,N-errors))
    rows.append(dict(case=case,distance=d,profile=profile,errors=errors,shots=N,ler=errors/N,ci95=[lo,hi]));print(case,d,profile,errors/N,flush=True)
 ROOT.joinpath('results/validation/physics_scaling_controls.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':run()
