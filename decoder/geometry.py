"""Physical-mechanism MA costs: exact d3 Jacobians, sampled MAP above d3."""
import numpy as np
from scipy.special import logsumexp
from physics import logical,draw
from panqec_distance_backend import _metric_closure


def closure(C):
    D=np.asarray(C,float).copy()
    for k in range(len(C)):D=np.minimum(D,D[:,k,None]+D[None,k,:])
    np.fill_diagonal(D,0);return D


def norm(C):
    v=C[C>1e-10];return C/max(np.median(v),1e-12) if len(v) else C


def costs(expert,cal,src,seed=314159,kappa=3.,witness_mode="disagreement",regret_reference="environment",reference_data=None,witness_sampling="nominal"):
    from scalable_geometry import base_precision,metric_costs
    # Imported from v8.9 complete module, copied without the GNN architecture.
    code,F,H,Z=expert.code,expert.F,expert.H,expert.Z;K=len(Z);M=Z.shape[1];A=base_precision(code,F,H)
    if expert.exact:
        idx=np.arange(1<<(code.n_stabilizers+2));shift=expert.shifts
        def make(p,mode):
            from physics import exact_tables
            tab=exact_tables(code,F,H,np.asarray([p]))[0][0];flat=tab.T.ravel();der=[]
            for rate,sh in zip(p,shift):
                # P=(1-p_j)Q+p_j Q xor sh; invert this 2x2 XOR kernel.
                Q=((1-rate)*flat-rate*flat[idx^sh])/max(1-2*rate,1e-10)
                der.append((Q[idx^sh]-Q).reshape(4,-1).T)
            der=np.asarray(der);order=np.argsort(tab,axis=1);aa=order[:,-1];bb=order[:,-2];r=np.arange(len(tab))
            h=np.maximum(p,.001)[None,:]*(der[:,r,aa].T/np.maximum(tab[r,aa,None],1e-20)-der[:,r,bb].T/np.maximum(tab[r,bb,None],1e-20))
            w=np.minimum(tab[r,aa],tab[r,bb]);keep=(w>1e-12)&(np.linalg.norm(h,axis=1)>1e-10);h=h[keep];w=w[keep];w/=w.sum()
            if mode=='paper':h/=np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-12)
            U=np.sqrt(w)[:,None]*h
            if mode=='legacy':U/=max(np.linalg.norm(U),1e-12)
            AU=np.asarray(A@U.T);B=np.linalg.inv(np.eye(len(U))+U@AU)
            trace=A.diagonal().sum()-np.trace(B@(AU.T@AU))
            def quadratic(delta):
                delta=delta/np.maximum(p,.001);v=AU.T@delta
                return max(0.,float(delta@(A@delta)-v@B@v)*M/max(trace,1e-12))
            return quadratic,int(len(U))
        def matrix(fn):
            C=np.zeros((K,K))
            for i in range(K):
                for j in range(i):C[i,j]=C[j,i]=np.sqrt(fn(Z[i]-Z[j]))
            return norm(C)
        p=src@Z;f,nw=make(p,'paper');Cp=matrix(f);Cl=matrix(make(p,'legacy')[0]);fn=[make(z,'paper')[0] for z in Z];Ce=np.zeros((K,K))
        for i in range(K):
            for j in range(i):v=Z[i]-Z[j];Ce[i,j]=Ce[j,i]=np.sqrt(.5*(fn[i](v)+fn[j](v)))
        Ce=norm(Ce);D=matrix(lambda d:np.sum((d/np.maximum(p,.001))**2));Cm=norm(closure(np.minimum(Cp,Ce)))
        cs={'euclid':D,'paper_norm':Cp,'legacy_regret_rawmag':Cl,'local_ma_regret':Ce,'multiscale_ma_regret':Cm}
    else:
        # Independent nominal anchors, without final test samples or labels.
        if witness_sampling=='nominal':anchors=draw(code,F,H,Z,src,12,8,seed+700000)
        elif witness_sampling=='balanced':
            parts=[draw(code,F,H,Z,np.eye(K)[k],6,8,seed+700000+1009*k) for k in range(K)]
            anchors={'S':np.concatenate([v['S'] for v in parts])}
        else:raise ValueError(witness_sampling)
        faults=[];classes=[]
        from physics import decode
        base=decode(code,expert.match,anchors['S'])
        coset_decoder=None
        if witness_mode=='coset':
            from scipy import sparse
            from ldpc_compat import BpOsdDecoder
            # Explicitly constrain raw effective logical bits as well as syndrome.
            LF=logical(code,F.toarray().T)
            LH=np.stack([LF&1,(LF>>1)&1]).astype(np.uint8)
            Hext=sparse.vstack([H,sparse.csr_matrix(LH)],format='csc')
            coset_decoder=BpOsdDecoder(Hext,error_channel=np.clip(src@Z,1e-9,.49).tolist(),max_iter=30,bp_method='minimum_sum',osd_method='OSD_0',osd_order=0,omp_thread_count=1)
        for s,bc in zip(anchors['S'],base):
            ff=[np.asarray(bp.decoder.decode(s),np.uint8).copy() for bp in expert.bp]
            if coset_decoder is not None:
                for cls in range(4):
                    rhs=np.r_[s,cls&1,(cls>>1)&1].astype(np.uint8)
                    f=np.asarray(coset_decoder.decode(rhs),np.uint8).copy()
                    assert np.array_equal(np.asarray(Hext@f).ravel()%2,rhs)
                    ff.append(f)
            # A syndrome-consistent MWPM candidate in grouped fault coordinates.
            f=np.zeros(M,np.uint8);f[:code.n]=bc[:code.n];f[2*code.n:3*code.n]=bc[code.n:];ff.append(f)
            for f in ff:assert np.array_equal(np.asarray(H@f).ravel()%2,s)
            cc=logical(code,np.asarray([np.asarray(F@f).ravel()%2 for f in ff],np.uint8));classes.append(cc^cc[0]);faults.append(np.stack(ff))
        cs,nw=metric_costs(Z,src,A,np.asarray(faults),np.asarray(classes))
    # Frozen source-calibration reference; soft-selector geometry uses validation.
    if regret_reference=='soft_selector':cal=reference_data
    from physics import hmm
    gp=hmm(cal['S'],cal['LL'],src,cal['nseq'],cal['rounds'],expert.history,expert.train_stay)
    pred=np.einsum('nk,nkl->nl',gp,cal['P']).argmax(1);known=cal['P'][np.arange(len(pred)),cal['E']].argmax(1)
    risk=np.array([(pred[cal['E']==k]!=cal['Y'][cal['E']==k]).mean() for k in range(K)])
    floor=np.array([(known[cal['E']==k]!=cal['Y'][cal['E']==k]).mean() for k in range(K)])
    if regret_reference=='selector':
        miss=(cal['P'].argmax(2)!=cal['Y'][:,None]).all(1)
        floor=np.array([miss[cal['E']==k].mean() for k in range(K)])
    elif regret_reference=='soft_selector':
        mix=np.einsum('nk,nkl->nl',gp,cal['P']);y=cal['Y'];best=cal['P'][np.arange(len(y)),:,y].max(1)
        a=-np.log(np.maximum(mix[np.arange(len(y)),y],1e-10));b=-np.log(np.maximum(best,1e-10))
        risk=np.array([a[cal['E']==k].mean() for k in range(K)]);floor=np.array([b[cal['E']==k].mean() for k in range(K)])
    reg=np.maximum(risk-floor,0);h=(reg-reg.min())/np.ptp(reg) if np.ptp(reg)>1e-12 else np.zeros(K)
    Ch=closure(cs['multiscale_ma_regret']/(1+kappa*.5*(h[:,None]+h[None,:])))
    result={'tail':np.ones((K,K))-np.eye(K),'euclid':cs['euclid'],'paper_ma':cs['paper_norm'],'legacy_ma':cs['legacy_regret_rawmag'],'local_ma':cs['local_ma_regret'],'multiscale_ma':cs['multiscale_ma_regret'],'hc':Ch}
    return result,dict(witness_count=nw,exact_jacobian=expert.exact,witness_mode=witness_mode,witness_sampling=witness_sampling,anchor_budget=256 if expert.exact else len(anchors['S']),reference_loss=regret_reference,reference_risk=floor.tolist(),source_hmm_risk=risk.tolist(),regret=reg.tolist(),hazard=h.tolist())
