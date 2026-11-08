from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(Path(__file__).parent),str(ROOT),str(ROOT/'scripts')]
from physics import *
from scipy import sparse
from ldpc_compat import BpOsdDecoder
from qecrobust.headline_specs import HEADLINE_CASES,HEADLINE_SOURCE_PI
for d in [3,5,7,15,19]:
 c,xy,ed,env,F,H,Z=setup(d,HEADLINE_CASES['hv_checker_cross_perimeter']);LF=logical(c,F.toarray().T);HH=sparse.vstack([H,sparse.csr_matrix(np.stack([LF&1,(LF>>1)&1]))],format='csc')
 dec=BpOsdDecoder(HH,error_channel=(np.asarray(HEADLINE_SOURCE_PI)@Z).tolist(),max_iter=30,bp_method='minimum_sum',osd_method='OSD_0',osd_order=0,omp_thread_count=1)
 data=draw(c,F,H,Z,np.ones(5)/5,3,2,771+d)
 for s in data['S']:
  corrections=[]
  for cls in range(4):
   rhs=np.r_[s,cls&1,(cls>>1)&1].astype(np.uint8);f=np.asarray(dec.decode(rhs),np.uint8);cor=np.asarray(F@f).ravel()%2
   assert np.array_equal(np.asarray(HH@f).ravel()%2,rhs) and logical(c,cor)==cls;corrections.append(cor)
  assert all(c.in_codespace(cor^corrections[0]) for cor in corrections)
 print('All four constrained logical cosets validated at distance',d,flush=True)
