from pathlib import Path
import sys,itertools,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(Path(__file__).parent),str(ROOT),str(ROOT/'scripts')]
from physics import *
ROOT.joinpath("results/validation").mkdir(parents=True,exist_ok=True)
from qecrobust.headline_specs import HEADLINE_CASES,HEADLINE_SOURCE_PI
out=[]
for d in [3,5,7]:
 c,xy,ed,env,F,H,Z=setup(d,HEADLINE_CASES['hv_checker_cross_perimeter']);n=c.n
 events=np.zeros((2*len(ed),2*n),np.uint8)
 for k,(a,b) in enumerate(ed):events[k,[a,b]]=1;events[k+len(ed),np.array([a,b])+n]=1
 dec=matching(c,env,ed,HEADLINE_SOURCE_PI)
 for w in [1,2]:
  combos=list(itertools.combinations(range(len(events)),w));count=0;sector={"XX_only":0,"ZZ_only":0,"mixed":0}
  for st in range(0,len(combos),4096):
   ids=np.array(combos[st:st+4096]);err=np.bitwise_xor.reduce(events[ids],axis=1);syn=np.asarray(c.measure_syndrome(err),np.uint8).T;co=decode(c,dec,syn);fail=logical(c,err^co)!=0;count+=int(np.sum(fail))
   for combo,bad in zip(ids,fail):
    if bad:sector['XX_only' if np.all(combo<len(ed)) else 'ZZ_only' if np.all(combo>=len(ed)) else 'mixed']+=1
  out.append(dict(distance=d,pair_events=w,combinations=len(combos),mwpm_logical_failures=count,by_sector=sector))
ROOT.joinpath('results/validation/pair_fault_audit.json').write_text(json.dumps(out,indent=2));print(out)
