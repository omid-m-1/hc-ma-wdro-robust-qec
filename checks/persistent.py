"""Persistent-space geometry/syndrome checks, not accuracy confirmation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'decoder'),str(ROOT/'scripts')]
import numpy as np
from qecrobust.case_bank import names,get_case
from physics import setup,draw,matching,decode,logical
for name in names('persistent'):
 bank=get_case(name)
 for d in [3,5]:
  code,xy,edges,env,F,H,Z=setup(d,bank['specs'],'matched',.6)
  assert len(Z)==5 and len(edges)==2*d*(d-1)
  data=draw(code,F,H,Z,np.ones(5)/5,10,4,107)
  correction=decode(code,matching(code,env,edges,bank['source']),data['S'])
  assert all(code.in_codespace(e^c) for e,c in zip(data['ERR'],correction))
  assert np.array_equal(data['S'],np.asarray(code.measure_syndrome(data['ERR']).T,np.uint8))
 print(name,'GEOMETRY/SYNDROME OK')
print('No control environment was added to the modeled support.')
