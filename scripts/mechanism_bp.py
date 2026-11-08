"""Joint X/Y/Z/XX/ZZ factor graph; BP+OSD is approximate, not min-weight/Bayes."""
import numpy as np
from scipy import sparse
from canonical_noise import mechanism_vector

class MechanismBP:
    def __init__(self,code,edges,envs,pi):
        from ldpc_compat import BpOsdDecoder
        n=code.n;ed=np.asarray(edges)
        eye=sparse.eye(n,dtype=np.uint8)
        edge=sparse.csc_matrix((np.ones(2*len(ed),np.uint8),
          (ed.ravel(),np.repeat(np.arange(len(ed)),2))),shape=(n,len(ed)))
        self.M=sparse.bmat([[eye,eye,None,edge,None],
                           [None,eye,eye,None,edge]],format='csc',dtype=np.uint8)
        if '_burst_matrix' in envs[0]:self.M=sparse.hstack([self.M,envs[0]['_burst_matrix']],format='csc',dtype=np.uint8)
        stab=code.stabilizer_matrix
        self.H=(sparse.hstack([stab[:,n:],stab[:,:n]])@self.M).tocsc()
        self.H.data%=2;self.H.eliminate_zeros()
        self.p=np.asarray(pi)@np.stack([mechanism_vector(e) for e in envs])
        self.decoder=BpOsdDecoder(self.H,error_channel=np.clip(self.p,1e-9,1-1e-9).tolist(),
          max_iter=30,bp_method='minimum_sum',osd_method='OSD_0',osd_order=0,omp_thread_count=1)
        self.cache={}
    def decode(self,s):
        s=np.asarray(s,np.uint8);key=np.packbits(s).tobytes()
        if key not in self.cache:
            z=np.asarray(self.decoder.decode(s),np.uint8)
            if np.any(np.asarray(self.H@z).ravel()%2!=s):raise RuntimeError('BP-OSD syndrome mismatch')
            self.cache[key]=np.asarray(self.M@z).ravel().astype(np.uint8)%2
        return self.cache[key]
    def check_probabilities(self):
        # Exact individual check marginals, but their product is only a
        # pseudolikelihood: correlated checks are NOT independent.
        H=self.H.tocsr();out=[]
        for i in range(H.shape[0]):
            out.append((1-np.prod(1-2*self.p[H.indices[H.indptr[i]:H.indptr[i+1]]]))/2)
        return np.clip(out,1e-8,1-1e-8)
