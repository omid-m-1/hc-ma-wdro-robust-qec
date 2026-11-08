"""Audit physically indistinguishable single faults; no learned/test labels."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'decoder'),str(ROOT),str(ROOT/'scripts')]
import numpy as np
from physics import setup,logical
from qecrobust.case_bank import get_case

def audit(d,geometry):
    c,xy,edges,env,F,H,Z=setup(d,get_case('band_family_topology')['specs'],noise_geometry=geometry)
    syn=H.toarray().T;labels=logical(c,F.toarray().T);p=Z[-1]
    zero=float(np.prod(1-p));single=zero*p/(1-p);groups={};pairs=0
    for i,s in enumerate(syn):
        groups.setdefault(np.packbits(s).tobytes(),np.zeros(4))[labels[i]]+=single[i]
        ids=np.flatnonzero(np.all(syn[i+1:]==s,axis=1))+i+1
        for j in ids:
            if labels[j]!=labels[i]:
                assert c.in_codespace((F[:,i].toarray().ravel()^F[:,j].toarray().ravel()).astype(np.uint8))
                pairs+=1
    groups.setdefault(bytes((H.shape[0]+7)//8),np.zeros(4))[0]+=zero
    # Restrict to zero/one-fault events. Even a regime-informed decoder cannot
    # beat this restricted-event loss, since observations are identical.
    bound=float(sum(a.sum()-a.max() for a in groups.values()))
    return dict(distance=d,geometry=geometry,length=env[-1]['burst_length'],alias_pairs=pairs,deployment_oracle_lower_bound=bound)

if __name__=='__main__':
    rows=[audit(d,g) for d,g in [(3,'long_segments'),(5,'long_segments'),(5,'segments'),(7,'long_segments')]]
    assert rows[1]['alias_pairs']>0 and rows[1]['deployment_oracle_lower_bound']>0
    assert rows[2]['alias_pairs']==0 and rows[2]['deployment_oracle_lower_bound']==0
    assert rows[3]['alias_pairs']==0
    for row in rows:print(row)
    print('Same-syndrome logical ambiguity checks passed; d5 short profile removes single-fault aliases.')
