"""Audited PanQEC/SciPy compatibility and physical-geometry helpers."""

from __future__ import annotations

import numpy as np

def _imports():
    # PanQEC 0.1.6 calls a removed private SciPy method. Preserve its exact
    # sparse assignment semantics through the public item-assignment API.
    from scipy.sparse import dok_matrix
    if not hasattr(dok_matrix,'_update'):
        def _update_compat(matrix, values):
            for key,value in values.items():
                matrix[key]=value
        dok_matrix._update=_update_compat
    try:
        from panqec.codes import RotatedPlanar2DCode
        from panqec.error_models import PauliErrorModel
        from panqec.decoders import MatchingDecoder
    except Exception as e:
        raise RuntimeError('PanQEC backend requires panqec (tested target: server panqec 0.1.x). Install with `python -m pip install panqec` or use distance=3 exact backend.') from e
    class CompatibleMatchingDecoder(MatchingDecoder):
        """Apply explicit weights with the supported PyMatching 2.x API."""
        def __init__(self, code, error_model, error_rate, error_type=None, weights=None):
            super().__init__(code,error_model,error_rate,error_type=error_type,weights=weights)
            from pymatching import Matching
            wx,wz=weights if weights is not None else error_model.get_weights(code,error_rate)
            if error_type is None or error_type == 'X':
                self.matcher_x=Matching(code.Hz,weights=wx)
            if error_type is None or error_type == 'Z':
                self.matcher_z=Matching(code.Hx,weights=wz)
    return RotatedPlanar2DCode,PauliErrorModel,CompatibleMatchingDecoder

def _norm_xy(coords):
    a=np.asarray([tuple(c)[:2] for c in coords],float)
    lo=a.min(0); hi=a.max(0); return (a-lo)/np.maximum(hi-lo,1e-9)

def _axis_ranks(v):
    u=np.unique(np.round(np.asarray(v,float),10))
    u=np.sort(u)
    return np.array([int(np.argmin(np.abs(u-x))) for x in v],dtype=int), max(len(u),1)

def _local_pattern(name,xy):
    x,y=xy[:,0],xy[:,1]
    ix,nx=_axis_ranks(x); iy,ny=_axis_ranks(y)
    w=np.ones(len(x),dtype=float)
    if name in ('uniform','none'): return w
    base=.85; hot=1.45; w[:]=base
    mx=(nx-1)//2; my=(ny-1)//2
    masks={
      'hotspot_left_col': ix==0,
      'hotspot_mid_col': ix==mx,
      'hotspot_right_col': ix==nx-1,
      'hotspot_top_row': iy==0,
      'hotspot_mid_row': iy==my,
      'hotspot_bottom_row': iy==ny-1,
      'cross': (ix==mx)|(iy==my),
      'checker_even': ((ix+iy)%2)==0,
      'checker_odd': ((ix+iy)%2)==1,
      'band_h_top': iy <= max(0,(ny-1)//2),
      'band_h_bottom': iy >= (ny-1)//2,
      'band_v_left': ix <= max(0,(nx-1)//2),
      'band_v_right': ix >= (nx-1)//2,
    }
    if name in masks:
        w[np.asarray(masks[name],bool)]=hot; return w
    if name.startswith('l_'):
        left=name in ('l_tl','l_bl'); top=name in ('l_tl','l_tr')
        bx=(ix<=max(0,(nx-1)//3)) if left else (ix>=max(0,2*(nx-1)//3))
        by=(iy<=max(0,(ny-1)//3)) if top else (iy>=max(0,2*(ny-1)//3))
        # L-shaped corner: boundary arm in x plus boundary arm in y.
        edge_x=(ix==0) if left else (ix==nx-1); edge_y=(iy==0) if top else (iy==ny-1)
        w[(bx & edge_y)|(by & edge_x)]=hot; return w
    if name=='gradient_lr': return .80+.65*x
    if name=='gradient_rl': return 1.45-.65*x
    if name=='gradient_tb': return .80+.65*y
    if name=='gradient_bt': return 1.45-.65*y
    return w

def _edges(xy):
    # O(n log n) nearest-neighbour construction.  v8.2 formed an O(n^2)
    # dense distance matrix, which makes large distances such as d=100
    # unnecessarily memory hungry.
    from scipy.spatial import cKDTree
    xy=np.asarray(xy,float); n=len(xy)
    if n<2: return []
    tree=cKDTree(xy); dist,_=tree.query(xy,k=2)
    d0=float(np.min(dist[:,1][dist[:,1]>1e-12]))
    pairs=tree.query_pairs(d0*1.05,output_type='ndarray')
    return np.asarray(pairs,dtype=np.int32).reshape(-1,2)

def _pair_pattern(name,xy,edges):
    if len(edges)==0: return np.zeros(0,float)
    pa=np.asarray([xy[i] for i,j in edges]); pb=np.asarray([xy[j] for i,j in edges]); mids=(pa+pb)/2
    dx=np.abs(pb[:,0]-pa[:,0]); dy=np.abs(pb[:,1]-pa[:,1])
    horiz=dx>=dy; vert=~horiz
    x,y=mids[:,0],mids[:,1]
    w=np.full(len(edges),.35,dtype=float)
    sel=np.zeros(len(edges),dtype=bool)
    if name in ('uniform','all'): return np.ones(len(edges),float)
    if name=='horizontal_all': sel=horiz
    elif name=='vertical_all': sel=vert
    elif name.startswith('horizontal_row'):
        r=int(name[-1]); target=[0.,.5,1.][r]; vals=np.abs(y[horiz]-target)
        if len(vals): sel=horiz & (np.abs(y-target)<=vals.min()+1e-9)
    elif name.startswith('vertical_col'):
        c=int(name[-1]); target=[0.,.5,1.][c]; vals=np.abs(x[vert]-target)
        if len(vals): sel=vert & (np.abs(x-target)<=vals.min()+1e-9)
    elif name=='perimeter':
        # Boundary edges only, preserving horizontal+vertical orientation.
        yh=np.minimum(np.abs(y-0.),np.abs(y-1.)); xv=np.minimum(np.abs(x-0.),np.abs(x-1.))
        mh=np.min(yh[horiz]) if np.any(horiz) else np.inf; mv=np.min(xv[vert]) if np.any(vert) else np.inf
        sel=(horiz & (yh<=mh+1e-9)) | (vert & (xv<=mv+1e-9))
    elif name=='center_cross':
        # Four edges nearest the lattice center: two horizontal, two vertical.
        dist=(x-.5)**2+(y-.5)**2
        for orient in (horiz,vert):
            ids=np.where(orient)[0]
            if len(ids): sel[ids[np.argsort(dist[ids])[:min(2,len(ids))]]]=True
    elif name=='mixed_a': sel=(horiz & (x<=.5)) | (vert & (x>.5))
    elif name=='mixed_b': sel=(horiz & (x>.5)) | (vert & (x<=.5))
    w[sel]=1.45
    return w

def _reference_pattern_mean(kind,name):
    """Mean multiplier of the frozen d=3 benchmark definition.

    Large-distance patterns are rescaled to these means so changing d does not
    silently change average local/pair noise strength.
    """
    try:
        if kind=='local':
            from patterns import local_pattern
            return float(np.mean(local_pattern(name)))
        from patterns import pair_pattern
        return float(np.mean(pair_pattern(name)))
    except Exception:
        return 1.0

def _match_reference_mean(w,target):
    w=np.asarray(w,float)
    m=float(np.mean(w)) if len(w) else 1.0
    return w*(float(target)/max(m,1e-12))

def _metric_closure(C):
    D=np.asarray(C,float).copy();np.fill_diagonal(D,0)
    for k in range(len(D)):D=np.minimum(D,D[:,k,None]+D[None,k,:])
    return D

def _robust_q(loss,nom,C,rho):
    from scipy.optimize import linprog
    K=len(nom);c=np.tile(-np.asarray(loss,float),K);Aeq=[]
    for i in range(K):
        r=np.zeros(K*K);r[i*K:(i+1)*K]=1;Aeq.append(r)
    sol=linprog(c,A_ub=np.asarray(C).reshape(1,-1),b_ub=[rho],A_eq=np.asarray(Aeq),b_eq=np.asarray(nom),bounds=(0,None),method='highs')
    if not sol.success:raise RuntimeError(sol.message)
    return sol.x.reshape(K,K).sum(0)
