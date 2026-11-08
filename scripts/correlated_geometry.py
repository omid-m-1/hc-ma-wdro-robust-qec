"""Explicit geometry stresses, fixed before evaluation; preserve endpoint budget.

ribbons: independent adjacent XX/ZZ events in narrow Gaussian H/V ribbons.
segments: move half the adjacent-pair endpoint budget to independent X/Z
segment events of length max(2,ceil(d/3)), using the same ribbon geometry.
This changes correlation length, NOT a fixed-local-noise distance-threshold test.
"""
import numpy as np
from scipy import sparse

def ribbon_weights(name,xy,locations):
 a=xy[np.asarray([v[0] for v in locations])];b=xy[np.asarray([v[-1] for v in locations])]
 x,y=((a+b)/2).T;h=np.abs(a[:,0]-b[:,0])>np.abs(a[:,1]-b[:,1]);v=~h
 sigma=1/18
 def bell(t,c):return np.exp(-.5*((t-c)/sigma)**2)
 if name in ('all','uniform'):return np.ones(len(locations))
 if name=='horizontal_all':return h.astype(float)
 if name=='vertical_all':return v.astype(float)
 if name.startswith('horizontal_row'):return h*bell(y,[1/6,.5,5/6][int(name[-1])])
 if name.startswith('vertical_col'):return v*bell(x,[1/6,.5,5/6][int(name[-1])])
 if name=='center_cross':return h*bell(y,.5)+v*bell(x,.5)
 if name=='perimeter':return h*np.maximum(bell(y,0),bell(y,1))+v*np.maximum(bell(x,0),bell(x,1))
 if name=='mixed_a':return h*(x<=.5)+v*(x>.5)
 if name=='mixed_b':return h*(x>.5)+v*(x<=.5)
 raise ValueError(name)

def distribute(budget,counts,raw,concentration=.6,cap=.18):
 # Sum p_j*support_size_j equals the requested endpoint budget exactly.
 counts=np.asarray(counts,float);raw=np.asarray(raw,float)
 base=budget/counts.sum();target=raw/max(float((counts*raw).sum()/counts.sum()),1e-15)
 max_target=float(target.max());eta=concentration
 if max_target>1:eta=min(eta,max(0.,(cap/max(base,1e-15)-1)/(max_target-1)))
 p=base*((1-eta)+eta*target)
 assert np.all(p>=0) and p.max()<=cap+1e-10
 assert np.isclose(p@counts,budget,rtol=1e-10,atol=1e-12)
 return p,float(eta)

def augment(code,xy,edges,F,env,variant):
 policy=variant
 if variant=='distance_scaled':variant='segments' if code.n==25 else 'long_segments'
 if variant=='standard':return F,env
 if variant not in ('ribbons','segments','long_segments','long_aligned'):raise ValueError(variant)
 n=code.n;d=int(round(np.sqrt(n)));ed=[list(e) for e in edges];long=variant in ('long_segments','long_aligned');L=max(2,int(np.ceil(d/(2 if long else 3))))
 segment=variant in ('segments','long_segments','long_aligned')
 locations=[]
 if segment:
  starts=list(range(0,d-L+1,L))
  if starts[-1]!=d-L:starts.append(d-L)
  for axis in (0,1):
   for row in sorted(np.unique(xy[:,1-axis])):
    ids=np.flatnonzero(np.isclose(xy[:,1-axis],row));ids=ids[np.argsort(xy[ids,axis])]
    for st in starts:locations.append(ids[st:st+L].tolist())
  rr=[];cc=[]
  for j,ids in enumerate(locations):rr.extend(ids);cc.extend([j]*len(ids))
  B=sparse.csc_matrix((np.ones(len(rr),np.uint8),(rr,cc)),shape=(n,len(locations)))
  zero=sparse.csc_matrix(B.shape,dtype=np.uint8)
  BF=sparse.bmat([[B,zero],[zero,B]],format='csc',dtype=np.uint8)
  F=sparse.hstack([F,BF],format='csc')
 for e in env:
  oldx=2*float(e['pxx'].sum());oldz=2*float(e['pzz'].sum());fraction=.5 if segment else 0.
  raw=ribbon_weights(e['pair'],xy,ed)
  e['pxx'],eta_x=distribute((1-fraction)*oldx,np.full(len(ed),2),raw)
  e['pzz'],eta_z=distribute((1-fraction)*oldz,np.full(len(ed),2),raw)
  e['geometry_policy']=policy;e['geometry_variant']=variant;e['primitive_pair_endpoint_budget']=[oldx,oldz];e['ribbon_concentration']=[eta_x,eta_z]
  if fraction:
   raw=ribbon_weights(e['pair'],xy,locations);counts=np.array([len(ids) for ids in locations])
   rx=raw;rz=raw
   if variant=='long_aligned':
    # Determine logical-support axes from physical stabilizer/logical actions.
    axes=[]
    for half in [0,1]:
     found=[]
     for axis in [0,1]:
      ids=np.flatnonzero(np.isclose(xy[:,1-axis],.5));err=np.zeros(2*n,np.uint8);err[half*n+ids]=1
      if code.in_codespace(err) and np.any(code.logical_errors(err)):found.append(axis)
     if len(found)!=1:raise RuntimeError('could not identify unique straight logical-support axis')
     axes.append(found[0])
    a=xy[np.asarray([ids[0] for ids in locations])];b=xy[np.asarray([ids[-1] for ids in locations])];orient=(np.abs(a[:,1]-b[:,1])>np.abs(a[:,0]-b[:,0])).astype(int)
    # Bias both ribbon foreground and background; keep support nonzero.
    pattern=.4+.6*raw/max(float((counts*raw).sum()/counts.sum()),1e-15)
    rx=pattern*np.where(orient==axes[0],4.,1.);rz=pattern*np.where(orient==axes[1],4.,1.)
    e['logical_axis_x'],e['logical_axis_z']=axes
   bx,beta_x=distribute(fraction*oldx,counts,rx,concentration=1. if variant=='long_aligned' else .6);bz,beta_z=distribute(fraction*oldz,counts,rz,concentration=1. if variant=='long_aligned' else .6)
   a=xy[np.asarray([ids[0] for ids in locations])];b=xy[np.asarray([ids[-1] for ids in locations])];horizontal=np.abs(a[:,0]-b[:,0])>np.abs(a[:,1]-b[:,1])
   e['burst_x_horizontal_share']=float((bx*counts*horizontal).sum()/max(bx@counts,1e-15));e['burst_z_horizontal_share']=float((bz*counts*horizontal).sum()/max(bz@counts,1e-15))
   e['_burst_matrix']=BF;e['_burst_prob']=np.r_[bx,bz];e['burst_length']=L;e['burst_concentration']=[beta_x,beta_z]
   assert np.isclose(2*e['pxx'].sum()+bx@counts,oldx)
   assert np.isclose(2*e['pzz'].sum()+bz@counts,oldz)
 return F,env
