"""Independent Bernoulli X/Y/Z/XX/ZZ faults, shared with exact d=3.

Coordinates follow the exact benchmark's row-major qubit indexing. Fractional
patterns preserve their spatial extent as d grows; `thin` preserves old widths.
No probability/rate is selected from final-test performance.
"""
import numpy as np
from panqec_distance_backend import _edges

BASE = {'pauli_local':(.020,0.), 'pauli_correlated':(.012,.025),
        'crosstalk_pauli':(.004,.035), 'mixed_pauli':(.014,.024)}

def coordinates(code):
    raw=np.asarray(code.qubit_coordinates,float)[:,:2]
    # PanQEC orders its coordinates first-axis major; the exact benchmark's
    # named horizontal direction increases the second physical coordinate.
    raw=raw[:,::-1]
    return (raw-raw.min(0))/np.maximum(np.ptp(raw,axis=0),1e-12)

def local_weights(name,xy,scaling='fractional'):
    from panqec_distance_backend import _local_pattern
    if scaling=='thin': return _local_pattern(name,xy)
    x,y=xy.T; n=int(round(np.sqrt(len(x)))); tol=1e-9
    left=x<1/3-tol;right=x>2/3+tol;top=y<1/3-tol;bottom=y>2/3+tol
    midx=(~left)&(~right);midy=(~top)&(~bottom)
    masks={'hotspot_left_col':left,'hotspot_mid_col':midx,'hotspot_right_col':right,
      'hotspot_top_row':top,'hotspot_mid_row':midy,'hotspot_bottom_row':bottom,
      'cross':midx|midy,'checker_even':((np.rint(x*(n-1))+np.rint(y*(n-1)))%2)==0,
      'checker_odd':((np.rint(x*(n-1))+np.rint(y*(n-1)))%2)==1,
      'band_h_top':y<=.5+tol,'band_h_bottom':y>=.5-tol,
      'band_v_left':x<=.5+tol,'band_v_right':x>=.5-tol,
      'l_tl':(top&(~right))|(left&(~bottom)),
      'l_tr':(top&(~left))|(right&(~bottom)),
      'l_bl':(bottom&(~right))|(left&(~top)),
      'l_br':(bottom&(~left))|(right&(~top)),
      'hotspot_tl_2x2':(x<=.5+tol)&(y<=.5+tol),
      'hotspot_tr_2x2':(x>=.5-tol)&(y<=.5+tol),
      'hotspot_bl_2x2':(x<=.5+tol)&(y>=.5-tol),
      'hotspot_br_2x2':(x>=.5-tol)&(y>=.5-tol),
      'diag_main':np.abs(x-y)<1/6+tol,'diag_anti':np.abs(x+y-1)<1/6+tol}
    if name in ('uniform','none'):return np.ones(len(x))
    if name in masks:return np.where(masks[name],1.45,.85)
    v={'gradient_lr':x,'gradient_rl':1-x,'gradient_tb':y,'gradient_bt':1-y}.get(name)
    if v is not None:return np.interp(v,[0,.5,1],[.80,1.10,1.45])
    raise ValueError(name)

def pair_weights(name,xy,edges,scaling='fractional'):
    from panqec_distance_backend import _pair_pattern
    if scaling=='thin':return _pair_pattern(name,xy,edges)
    e=np.asarray(edges);a=xy[e[:,0]];b=xy[e[:,1]];x,y=((a+b)/2).T
    horizontal=np.abs(a[:,0]-b[:,0])>np.abs(a[:,1]-b[:,1]);vertical=~horizontal
    if name in ('uniform','all'):return np.ones(len(e))
    if name=='horizontal_all':sel=horizontal
    elif name=='vertical_all':sel=vertical
    elif name.startswith('horizontal_row'):
        k=int(name[-1]);sel=horizontal & ([y<1/3,(y>=1/3)&(y<=2/3),y>2/3][k])
    elif name.startswith('vertical_col'):
        k=int(name[-1]);sel=vertical & ([x<1/3,(x>=1/3)&(x<=2/3),x>2/3][k])
    elif name=='perimeter':sel=(horizontal&((y<.25)|(y>.75)))|(vertical&((x<.25)|(x>.75)))
    elif name=='center_cross':sel=(horizontal&(np.abs(y-.5)<=1/6+1e-9))|(vertical&(np.abs(x-.5)<=1/6+1e-9))
    elif name=='mixed_a':sel=(horizontal&(x<=.5))|(vertical&(x>.5))
    elif name=='mixed_b':sel=(horizontal&(x>.5))|(vertical&(x<.5))
    else:raise ValueError(name)
    return np.where(sel,1.45,.35)

def environment(spec,xy,edges,scaling='fractional'):
    fam,local,pair,severity=spec
    l=local_weights(local,xy,scaling);p=pair_weights(pair,xy,edges,scaling)
    # Fix mean strength across distance; only the relative spatial allocation
    # changes. This is an experimental definition, not a fit to test LER.
    from patterns import local_pattern,pair_pattern
    l=l*(float(local_pattern(local).mean())/l.mean())
    p=p*(float(pair_pattern(pair).mean())/p.mean())
    a,b=BASE[fam];a*=severity;b*=severity
    env=dict(family=fam,local=local,pair=pair,px=np.clip(.85*a*l,0,.22),
      py=np.clip(.30*a*l,0,.22),pz=np.clip(a*l,0,.22),
      pxx=np.clip(.75*b*p,0,.22),pzz=np.clip(b*p,0,.22),
      local_multiplier_mean=float(l.mean()),pair_multiplier_mean=float(p.mean()))
    return env

def sample_error(code,env,edges,rng):
    n=code.n;e=np.zeros(2*n,np.uint8)
    bx=rng.random(n)<env['px'];by=rng.random(n)<env['py'];bz=rng.random(n)<env['pz']
    e[:n]=bx^by;e[n:]=bz^by
    ed=np.asarray(edges)
    for half,key in [(e[:n],'pxx'),(e[n:],'pzz')]:
        fired=ed[rng.random(len(ed))<env[key]]
        if len(fired):np.bitwise_xor.at(half,fired.ravel(),1)
    if '_burst_matrix' in env:e^=np.asarray(env['_burst_matrix']@(rng.random(len(env['_burst_prob']))<env['_burst_prob'])).ravel().astype(np.uint8)%2
    return e

def marginals(env,edges):
    mx=env['px']+env['py']-2*env['px']*env['py']
    mz=env['pz']+env['py']-2*env['pz']*env['py']
    for (i,j),px,pz in zip(edges,env['pxx'],env['pzz']):
        for v in (i,j):
            mx[v]=mx[v]+px-2*mx[v]*px
            mz[v]=mz[v]+pz-2*mz[v]*pz
    if '_burst_matrix' in env:
        B=env['_burst_matrix'].tocsc();n=len(mx)
        for j,p in enumerate(env['_burst_prob']):
            for q in B.indices[B.indptr[j]:B.indptr[j+1]]:
                half=mx if q<n else mz;v=q if q<n else q-n;half[v]=half[v]+p-2*half[v]*p
    return mx,mz

def mechanism_vector(env):
    primitive=np.r_[env['px'],env['py'],env['pz'],env['pxx'],env['pzz']]
    return np.r_[primitive,env['_burst_prob']] if '_burst_prob' in env else primitive
