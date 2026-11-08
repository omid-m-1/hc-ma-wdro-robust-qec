"""Case/support-definition checks; no learner fitting."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qecrobust.case_bank import names,get_case
assert len(names('hv'))==4 and len(names('persistent'))==7 and len(names('all'))==11
for name in names():
 c=get_case(name)
 assert len(c['specs'])==len(c['source'])==len(c['target'])==5
 assert abs(sum(c['source'])-1)<1e-12 and abs(sum(c['target'])-1)<1e-12
 assert c['modeled_indices']==[0,1,2,3,4]
 assert c['source'][4]==0 and c['target'][4]==.93
 if c['family']=='persistent':assert c['control'] is not None and c['control'] not in c['specs']
print('CASE BANK AND EXCLUDED-CONTROL DEFINITIONS OK')
