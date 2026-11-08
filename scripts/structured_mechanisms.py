"""Independent Pauli-fault model, identical rates and fault semantics at every d."""
import numpy as np
from scipy import sparse
from panqec_distance_backend import _imports,_norm_xy,_edges,_local_pattern,_pair_pattern,_match_reference_mean,_reference_pattern_mean

def topology(d):
 C,_,_=_imports();code=C(d,d);xy=_norm_xy(code.qubit_coordinates);edges=_edges(xy);n=code.n
 rr=[];cc=[]
 for q in range(n):
  for t in range(3):
   col=3*q+t
   if t in (0,1):rr.append(q);cc.append(col)
   if t in (1,2):rr.append(n+q);cc.append(col)
 for j,(a,b) in enumerate(edges):
  for t in range(2):
   col=3*n+2*j+t
   for q in (a,b):rr.append(int(q)+t*n);cc.append(col)
 F=sparse.csc_matrix((np.ones(len(rr),np.uint8),(rr,cc)),shape=(2*n,3*n+2*len(edges)))
 S=code.stabilizer_matrix.tocsr().astype(np.uint8)
 H=(sparse.hstack([S[:,n:],S[:,:n]],format='csr')@F).astype(np.uint8);H.data%=2;H.eliminate_zeros()
 A=(S[:,:n]+S[:,n:]).astype(np.float32)
 return code,xy,edges,F,H,A

def environment(spec,xy,edges):
 fam,local,pair,severity=spec
 localw=_match_reference_mean(_local_pattern(local,xy),_reference_pattern_mean('local',local))
 pairw=_match_reference_mean(_pair_pattern(pair,xy,edges),_reference_pattern_mean('pair',pair))
 rates={'pauli_local':(.020,0.),'pauli_correlated':(.012,.025),'crosstalk_pauli':(.004,.035),'mixed_pauli':(.014,.024)}
 bl,bp=rates[fam]
 z=np.concatenate([(bl*severity*localw[:,None]*np.array([.85,.30,1.])).ravel(),(bp*severity*pairw[:,None]*np.array([.75,1.])).ravel()])
 return np.clip(z,0,.22)

def marginal(F,p):
 # XOR of independent Bernoulli faults; exact one-bit marginal.
 return .5*(1-np.exp(F@np.log(np.clip(1-2*np.asarray(p),1e-12,1))))

def pauli_marginals(F,p,n):
 px=marginal(F[:n],p);pz=marginal(F[n:],p)
 # E[(-1)^(X xor Z)] gives the full four-class single-qubit marginal.
 Q=(F[:n]+F[n:]).astype(np.uint8);Q.data%=2;Q.eliminate_zeros();pc=marginal(Q,p)
 a=1-2*px;b=1-2*pz;c=1-2*pc
 return np.stack([(1+a+b+c)/4,(1-a+b-c)/4,(1+a-b-c)/4,(1-a-b+c)/4],1)

def sample(code,F,H,Z,pi,nseq,rounds,seed,stay=.95,history=8):
 rng=np.random.default_rng(seed);pi=np.asarray(pi,float);K=len(pi);T=np.tile((1-stay)*pi,(K,1));T[np.arange(K),np.arange(K)]+=stay
 E=[]
 for _ in range(nseq):
  k=rng.choice(K,p=pi)
  for t in range(rounds):
   if t:k=rng.choice(K,p=T[k])
   E.append(k)
 E=np.asarray(E,int);fault=(rng.random((len(E),F.shape[1]))<Z[E]).astype(np.uint8)
 ERR=np.asarray((F@fault.T).T,np.uint8)%2;S=np.asarray((H@fault.T).T,np.uint8)%2
 hist=S.reshape(nseq,rounds,-1).astype(np.float32);means=np.empty_like(hist)
 for t in range(rounds):means[:,t]=hist[:,max(0,t-history+1):t+1].mean(1)
 return {'S':S,'history_mean':means.reshape(S.shape),'E':E,'ERR':ERR,'nseq':nseq,'rounds':rounds}

def logical(code,e):
 e=np.asarray(e,np.uint8);le=np.asarray(code.logical_errors(e),np.uint8).reshape(-1)
 return int(le[0])|(int(le[1])<<1)

def matching(code,px,pz):
 from pymatching import Matching
 px=np.clip(px,1e-5,1-1e-5);pz=np.clip(pz,1e-5,1-1e-5)
 return Matching(code.Hz,weights=np.log((1-px)/px)),Matching(code.Hx,weights=np.log((1-pz)/pz))

def match_one(code,dec,s):
 return np.concatenate([dec[0].decode(code.extract_z_syndrome(s)),dec[1].decode(code.extract_x_syndrome(s))]).astype(np.uint8)

def decode_weights(code,probs,s,blend=1.,base_weights=None):
 px=probs[:,1]+probs[:,3];pz=probs[:,2]+probs[:,3]
 wx=np.log((1-np.clip(px,1e-5,1-1e-5))/np.clip(px,1e-5,1-1e-5));wz=np.log((1-np.clip(pz,1e-5,1-1e-5))/np.clip(pz,1e-5,1-1e-5))
 if base_weights is not None:wx=(1-blend)*base_weights[0]+blend*wx;wz=(1-blend)*base_weights[1]+blend*wz
 from pymatching import Matching
 return np.concatenate([Matching(code.Hz,weights=wx).decode(code.extract_z_syndrome(s)),Matching(code.Hx,weights=wz).decode(code.extract_x_syndrome(s))]).astype(np.uint8)

def bp_decoder(H,p):
 try:
  from ldpc_compat import BpOsdDecoder
 except ImportError as exc:
  raise RuntimeError('The structured benchmark requires ldpc>=2.2; install requirements-v89.txt') from exc
 return BpOsdDecoder(H,error_channel=np.clip(p,1e-6,.49).tolist(),max_iter=30,bp_method='minimum_sum',ms_scaling_factor=.625,osd_method='OSD_0',osd_order=0,omp_thread_count=1)
