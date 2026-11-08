"""Version 9.0 audited physical models and logical conventions."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
from scipy import sparse
from scipy.special import logsumexp
from panqec_distance_backend import _imports,_edges
from canonical_noise import coordinates,environment as original_environment,local_weights,pair_weights,marginals
from mechanism_bp import MechanismBP


def logical(code,err):
    # Action-tested, not inferred from PanQEC's misleading docstring.
    x=np.asarray(code.logical_errors(err),np.uint8)
    return x[...,0].astype(int)+2*x[...,1].astype(int)


def setup(d,specs,profile='matched',strength=.6,noise_geometry='standard'):
    code=_imports()[0](d);xy=coordinates(code);edges=_edges(xy);env=[]
    if profile not in ('original','matched'):raise ValueError(profile)
    for spec in specs:
        e=original_environment(spec,xy,edges,'fractional')
        if profile=='matched':
            # Self-similar masks are from canonical_noise. Hold mean local rate
            # and expected pair-fault endpoints per data qubit at d=3 values.
            for k in ('px','py','pz'):e[k]*=strength
            exposure_ratio=(2*12/9)/(2*len(edges)/code.n)
            for k in ('pxx','pzz'):e[k]*=strength*exposure_ratio
        e['profile']=profile;e['strength']=1. if profile=='original' else strength
        env.append(e)
    n=code.n;eye=sparse.eye(n,dtype=np.uint8);ed=np.asarray(edges)
    edge=sparse.csc_matrix((np.ones(2*len(ed),np.uint8),(ed.ravel(),np.repeat(np.arange(len(ed)),2))),shape=(n,len(ed)))
    F=sparse.bmat([[eye,eye,None,edge,None],[None,eye,eye,None,edge]],format='csc',dtype=np.uint8)
    from correlated_geometry import augment
    F,env=augment(code,xy,edges,F,env,noise_geometry)
    S=code.stabilizer_matrix.tocsr().astype(np.uint8)
    H=(sparse.hstack([S[:,n:],S[:,:n]])@F).astype(np.uint8);H.data%=2;H.eliminate_zeros()
    from canonical_noise import mechanism_vector
    Z=np.stack([mechanism_vector(e) for e in env])
    return code,xy,edges,env,F,H,Z


def matching(code,env,edges,pi):
    from pymatching import Matching
    mm=[marginals(e,edges) for e in env];px=sum(p*m[0] for p,m in zip(pi,mm));pz=sum(p*m[1] for p,m in zip(pi,mm))
    px=np.clip(px,1e-8,.49999);pz=np.clip(pz,1e-8,.49999)
    return (Matching(code.Hz,weights=np.log((1-px)/px)),Matching(code.Hx,weights=np.log((1-pz)/pz)))


def decode(code,matcher,S):
    S=np.asarray(S,np.uint8).reshape(-1,code.n_stabilizers)
    # PanQEC has zero rows in each CSS half; use its explicit indices.
    a=matcher[0].decode_batch(S[:,code.z_indices]);b=matcher[1].decode_batch(S[:,code.x_indices])
    return np.concatenate([a,b],axis=1).astype(np.uint8)


def draw(code,F,H,Z,pi,nseq,rounds,seed,stay=.95):
    rng=np.random.default_rng(seed);pi=np.asarray(pi,float);K=len(pi);E=np.empty((nseq,rounds),int)
    E[:,0]=rng.choice(K,nseq,p=pi)
    for t in range(1,rounds):
        keep=rng.random(nseq)<stay;E[:,t]=np.where(keep,E[:,t-1],rng.choice(K,nseq,p=pi))
    e=E.ravel();fault=(rng.random((len(e),F.shape[1]))<Z[e]).astype(np.uint8)
    err=np.asarray((F@fault.T).T,np.uint8)%2;S=np.asarray((H@fault.T).T,np.uint8)%2
    return dict(S=S,ERR=err,E=e,nseq=nseq,rounds=rounds,pi=pi,stay=stay)


def history_indices(nseq,rounds,history):
    t=np.arange(rounds)[:,None]-np.arange(history-1,-1,-1)[None,:]
    valid=t>=0;idx=np.maximum(t,0)[None,:,:]+np.arange(nseq)[:,None,None]*rounds
    return idx.reshape(-1,history),np.tile(valid,(nseq,1))


def exact_tables(code,F,H,Z):
    if code.n!=9:raise ValueError('exact enumeration only d=3')
    m=code.n_stabilizers;ss=np.asarray(H.toarray(),int).T@(1<<np.arange(m));ll=logical(code,F.toarray().T)
    shift=ss+(ll<<m);N=1<<(m+2);idx=np.arange(N);tabs=[]
    for z in Z:
        p=np.zeros(N);p[0]=1
        for rate,sh in zip(z,shift):p=(1-rate)*p+rate*p[idx^sh]
        tabs.append(p.reshape(4,1<<m).T)
    return np.asarray(tabs),shift


def hmm(S,emissions,pi,nseq,rounds,history,stay):
    idx,valid=history_indices(nseq,rounds,history);L=np.asarray(emissions)
    K=len(pi);T=stay*np.eye(K)+(1-stay)*np.tile(pi,(K,1));post=np.tile(pi,(len(idx),1)).astype(float)
    # Each window has stationary initial pi; padded steps leave it unchanged.
    for j in range(history):
        v=valid[:,j];prev=j>0 and valid[:,j-1]
        if j>0:
            mask=v&prev;post[mask]=post[mask]@T
        rows=np.flatnonzero(v);loglike=L[idx[rows,j]];like=np.exp(loglike-loglike.max(1,keepdims=True));post[rows]*=like
        post[rows]/=np.maximum(post[rows].sum(1,keepdims=True),1e-300)
    return post


def component_minweight_prediction(code,F,H,z,S):
    """Exact CSS component min-weight mechanism DP (the original Corr-Min objective)."""
    out=np.zeros(len(S),int)
    for bit,rows in [(0,code.z_indices),(1,code.x_indices)]:
        rows=np.flatnonzero(rows) if np.asarray(rows).dtype==bool else np.asarray(rows)
        h=H[rows].toarray().T.astype(int);m=len(rows);part=F.toarray().T.copy()
        if bit==0:part[:,code.n:]=0
        else:part[:,:code.n]=0
        lab=(logical(code,part)>>bit)&1;effect=h@(1<<np.arange(m))+(lab<<m)
        N=1<<(m+1);dp=np.full(N,np.inf);dp[0]=0;states=np.arange(N)
        for rate,shift in zip(z,effect):
            if rate>0 and shift:dp=np.minimum(dp,dp[states^shift]+np.log((1-rate)/rate))
        sid=S[:,rows].astype(int)@(1<<np.arange(m));pr=dp[sid+(1<<m)]<dp[sid]
        out|=pr.astype(int)<<bit
    return out


class Experts:
    """Exact d3 conditional cosets; empirical BP-OSD error-channel approximation above d3."""
    def __init__(self,code,env,edges,F,H,Z,src,expert_calibration="unconditional",history=8,train_stay=.95,test_stay=.98):
        self.history=history;self.train_stay=train_stay;self.test_stay=test_stay
        self.expert_calibration=expert_calibration;self.action_conf=None
        self.code=code;self.env=env;self.edges=edges;self.F=F;self.H=H;self.Z=Z;self.src=src;self.K=len(src)
        self.match=matching(code,env,edges,src);self.exact=code.n==9
        self.bp=[MechanismBP(code,edges,env,np.eye(self.K)[k]) for k in range(self.K)]
        self.sourcebp=MechanismBP(code,edges,env,src)
        self.conf=np.full((self.K,4),.25)
        if self.exact:
            self.tables,self.shifts=exact_tables(code,F,H,Z)
            self.mass=self.tables.sum(2);self.cond=self.tables/np.maximum(self.mass[:,:,None],1e-300)
            self.source_table=exact_tables(code,F,H,np.asarray([src@Z]))[0][0]
        self.checkp=np.clip(np.asarray([(1-np.exp(H@np.log(1-2*z)))/2 for z in Z]),1e-7,1-1e-7)
    def enrich(self,data,calibrate=False):
        code=self.code;S=data['S'];base=decode(code,self.match,S);labs=logical(code,data['ERR']^base)
        # Full symplectic logical projection is linear even before syndrome removal.
        bc=logical(code,base);data.update(Y=labs,base_label=bc)
        if self.exact:
            sid=S.astype(int)@(1<<np.arange(code.n_stabilizers));data['sid']=sid
            data['P']=self.cond[:,sid[:,None],np.arange(4)[None,:]^bc[:,None]].transpose(1,0,2)
            data['LL']=np.log(np.maximum(self.mass[:,sid].T,1e-300))
            data['actions']=data['P'].argmax(2)
        else:
            actions=np.empty((len(S),self.K),np.uint8)
            for k,bp in enumerate(self.bp):
                solutions=[]
                for st in range(0,len(S),2000):
                    solutions.extend(bp.decode(s) for s in S[st:st+2000])
                    if len(S)>=2000:print(f'BP+OSD candidate {k+1}/{self.K}: {min(st+2000,len(S))}/{len(S)} syndromes',flush=True)
                cor=np.asarray(solutions);actions[:,k]=logical(code,cor^base)
            if calibrate:
                for k in range(self.K):
                    ids=data['E']==k;res=labs[ids]^actions[ids,k]
                    count=np.bincount(res,minlength=4)+.5;self.conf[k]=count/count.sum()
                self.action_conf=np.empty((self.K,4,4))
                for k in range(self.K):
                    for a in range(4):
                        ids=(data['E']==k)&(actions[:,k]==a)
                        # Shrink rare proposal bins toward the calibrated BP error channel.
                        prior=self.conf[k][np.arange(4)^a]
                        count=np.bincount(labs[ids],minlength=4)+8*prior
                        self.action_conf[k,a]=count/count.sum()
            data['actions']=actions
            if self.expert_calibration=='action':
                data['P']=np.stack([self.action_conf[k,actions[:,k]] for k in range(self.K)],axis=1)
            else:
                data['P']=np.stack([self.conf[k][actions[:,k,None]^np.arange(4)[None,:]] for k in range(self.K)],axis=1)
            # Single-check product is a pseudolikelihood, not exact syndrome evidence.
            data['LL']=(S@np.log(self.checkp).T+(1-S)@np.log1p(-self.checkp).T)*.25
        # Deployment features depend solely on syndrome and known candidate models.
        # Centered log evidence is divided by sqrt(number of checks), retaining
        # scale while controlling large-distance optimization conditioning.
        LL=data['LL'];feat=(LL-LL.mean(1,keepdims=True))/max(np.sqrt(code.n_stabilizers),1)
        feat=np.concatenate([np.clip(feat,-12,12),S.mean(1,keepdims=True)],1).astype(np.float32)
        idx,valid=history_indices(data['nseq'],data['rounds'],self.history)
        x=feat[idx];x[~valid]=0;x=np.concatenate([x,valid[:,:,None]],2)
        data['X']=x;data['path']=data['E'][idx];data['valid']=valid
        # Every evaluated correction must actually remove the syndrome.
        assert all(code.in_codespace(e^c) for e,c in zip(data['ERR'][:32],base[:32]))
        assert np.array_equal(labs[:32]!=0,np.any(code.logical_errors((data['ERR']^base)[:32]),axis=1))
        return data
    def references(self,data):
        Y=data['Y'];S=data['S'];bc=data['base_label'];K=self.K;P=data['P']
        srcp=hmm(S,data['LL'],self.src,data['nseq'],data['rounds'],self.history,self.train_stay)
        tgt=data['target_pi'];tgtp=hmm(S,data['LL'],tgt,data['nseq'],data['rounds'],self.history,self.test_stay)
        srcpred=np.einsum('nk,nkl->nl',srcp,P).argmax(1);tgtpred=np.einsum('nk,nkl->nl',tgtp,P).argmax(1)
        if self.exact:
            sid=data['sid'];corrpred=component_minweight_prediction(self.code,self.F,self.H,self.src@self.Z,S)^bc
        else:
            cor=np.asarray([self.sourcebp.decode(s) for s in S]);corrpred=logical(self.code,cor)^bc
        known=P[np.arange(len(Y)),data['E']].argmax(1)
        return dict(mwpm=Y!=0,corr_min=corrpred!=Y,source_hmm=srcpred!=Y,bayes=tgtpred!=Y if self.exact else None,target_hmm_approx=tgtpred!=Y if not self.exact else None,known_environment=known!=Y)
