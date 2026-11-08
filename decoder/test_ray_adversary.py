from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from learning import adversary,path_weights,transport_distance
# An alternating path has zero first derivative at this source vertex but
# positive risk along a feasible ray. This reproduces a local-boundary trap.
path=np.array([[0,1,0,1]]);valid=np.ones_like(path,bool);loss=np.array([1.]);src=np.array([1.,0.]);C=np.ones((2,2))-np.eye(2)
a=adversary(loss,path,valid,src,C,.5,search='local');b=adversary(loss,path,valid,src,C,.5,search='rays')
fa=float(path_weights(a,path,valid).mean());fb=float(path_weights(b,path,valid).mean())
assert fb>fa+1e-10 and transport_distance(b,src,C)<=.5+1e-7
print('Boundary-trap test passed:',a,fa,'->',b,fb)
