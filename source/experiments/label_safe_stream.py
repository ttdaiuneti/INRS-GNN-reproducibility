"""Additive label-safe insertion validation; old experiment artifacts are untouched."""
from pathlib import Path
import sys, json, hashlib, datetime, csv
import numpy as np
import scipy.sparse as sp
import torch
H=Path(__file__).resolve().parent
R=H.parent
sys.path[:0]=[str(R),str(R/'shared')]
from core.data import load_planetoid_pyg
from core.gnn import GCN, train_gcn, set_seed
from core.rough_adjacency import incremental_rough_adjacency_update_sparse, batch_rough_adjacency_edgelist

def normalized(a):
    a=sp.csr_matrix(a,dtype=np.float64)+sp.eye(a.shape[0],format='csr')
    d=1/np.sqrt(np.asarray(a.sum(axis=1)).ravel())
    a=(sp.diags(d)@a@sp.diags(d)).tocoo()
    return torch.sparse_coo_tensor(np.vstack((a.row,a.col)),a.data.astype(np.float32),a.shape).coalesce()

def batch_radius(x, labels, trusted):
    out=np.full(len(x),np.inf)
    ix=np.flatnonzero(trusted)
    for v in ix:
        enemies=ix[labels[ix]!=labels[v]]
        if len(enemies): out[v]=np.linalg.norm(x[enemies]-x[v],axis=1).min()
    return out

def insert_radius(x, labels, trusted, delta, v):
    shrink=[]
    if trusted[v]:
        enemies=np.flatnonzero(trusted[:v] & (labels[:v]!=labels[v]))
        if len(enemies):
            ds=np.linalg.norm(x[enemies]-x[v],axis=1)
            shrink=enemies[ds<delta[enemies]].tolist()
            delta[enemies]=np.minimum(delta[enemies],ds)
            delta[v]=ds.min()
    return shrink

def rebuild(x,labels,delta,a):
    i,j,w=batch_rough_adjacency_edgelist(x,labels,delta,a,np.arange(x.shape[1]))
    return sp.csr_matrix((np.r_[w,w],(np.r_[i,j],np.r_[j,i])),shape=a.shape)

def error(a,b):
    assert np.array_equal(np.isinf(a),np.isinf(b))
    assert not np.isnan(a).any() and not np.isnan(b).any()
    m=np.isfinite(a)
    return float(np.max(np.abs(a[m]-b[m]),initial=0))

def edge_cases():
    cases=[]
    for seed in range(12):
        rng=np.random.default_rng(seed)
        x=rng.normal(size=(32,7));x[1]=x[0]
        labels=rng.integers(0,3,32);labels[:3]=[0,1,2]
        trusted=rng.random(32)>.35
        if seed==0: trusted[:]=False
        if seed==1: labels[:]=0;trusted[:]=True
        if seed==2: trusted[:]=True
        delta=np.full(32,np.inf)
        for v in range(32):
            insert_radius(x,labels,trusted,delta,v)
            assert error(delta[:v+1],batch_radius(x[:v+1],labels[:v+1],trusted[:v+1]))<=1e-12
        cases.append({'seed':seed,'passed':True})
    return cases

def run(name,seed,p,out):
    X,y,A,tr,va,te,nc=load_planetoid_pyg(name)
    rng=np.random.default_rng(seed)
    mandatory=np.flatnonzero(tr|va)
    rest=rng.permutation(np.flatnonzero(~(tr|va)))
    n0=int(len(y)*p['initial_fraction'])
    order=np.r_[mandatory,rest]
    X=np.asarray(X[order],dtype=np.float64);y=y[order];tr=tr[order];va=va[order]
    A=sp.csr_matrix(A[np.ix_(order,order)])
    # Fixed training/validation split; no held-out ground truth enters training tensors.
    safe_y=np.where(tr|va,y,0)
    torch.set_num_threads(1);set_seed(seed)
    model=GCN(X.shape[1],16,nc)
    train_gcn(model,torch.tensor(X[:n0],dtype=torch.float32),normalized(A[:n0,:n0]),torch.tensor(safe_y[:n0]),torch.tensor(tr[:n0]),torch.tensor(va[:n0]),torch.device('cpu'))
    model.eval()
    with torch.no_grad():
        probs=model(torch.tensor(X[:n0],dtype=torch.float32),normalized(A[:n0,:n0])).softmax(1)
    conf,pred=probs.max(1)
    labels=np.full(len(X),-1,dtype=int);trusted=np.zeros(len(X),bool)
    labels[:n0]=pred.numpy();trusted[:n0]=conf.numpy()>=p['tau']
    labels[:n0][tr[:n0]]=y[:n0][tr[:n0]];trusted[:n0][tr[:n0]]=True
    # Functions maintaining the operator receive only assigned labels, not y.
    del y,safe_y
    delta=np.full(len(X),np.inf);delta[:n0]=batch_radius(X[:n0],labels[:n0],trusted[:n0])
    W=sp.lil_matrix(A.shape,dtype=np.float64)
    W[:n0,:n0]=rebuild(X[:n0],labels[:n0],delta[:n0],A[:n0,:n0])
    records=[]
    for step in range(p['insertions']):
        v=n0+step;n=v+1
        with torch.no_grad():
            prob=model(torch.tensor(X[:n],dtype=torch.float32),normalized(A[:n,:n])).softmax(1)[v]
        labels[v]=int(prob.argmax());trusted[v]=float(prob.max())>=p['tau']
        shrink=insert_radius(X,labels,trusted,delta,v)
        touch={v}|set(A.indices[A.indptr[v]:A.indptr[v+1]][A.indices[A.indptr[v]:A.indptr[v+1]]<v])
        active=np.arange(len(X))<n
        updates,tadj=incremental_rough_adjacency_update_sparse(X,labels,delta,A,np.arange(X.shape[1]),touch,shrink,active_mask=active)
        for (i,j),w in updates.items():W[i,j]=w
        refd=batch_radius(X[:n],labels[:n],trusted[:n])
        refw=rebuild(X[:n],labels[:n],refd,A[:n,:n])
        diff=W.tocsr()[:n,:n]-refw
        de=error(delta[:n],refd);we=float(np.max(np.abs(diff.data),initial=0))
        assert de<=p['tolerance'] and we<=p['tolerance']
        records.append(dict(dataset=name,seed=seed,step=step+1,n_active=n,trusted_arrival=bool(trusted[v]),shrink=len(shrink),touch=len(tadj),delta_error=de,weight_error=we,passed=True))
    with torch.no_grad():
        z1=model(torch.tensor(X[:n],dtype=torch.float32),normalized(A[:n,:n]+W.tocsr()[:n,:n]))
        z2=model(torch.tensor(X[:n],dtype=torch.float32),normalized(A[:n,:n]+refw))
    logit_error=float((z1-z2).abs().max());assert logit_error<=1e-6
    summary={'dataset':name,'seed':seed,'initial_trusted':int(trusted[:n0].sum()),'trusted_arrivals':sum(r['trusted_arrival'] for r in records),'max_delta_error':max(r['delta_error'] for r in records),'max_weight_error':max(r['weight_error'] for r in records),'final_logit_error':logit_error,'n_rough_edges':int(sp.triu(refw).count_nonzero())}
    (out/f'{name}_{seed}.json').write_text(json.dumps({'summary':summary,'records':records},indent=2))
    print(summary,flush=True)
    return summary

def main():
    p=json.loads((H/'protocol.json').read_text());out=H/'results';out.mkdir(exist_ok=False)
    files=[H/'protocol.json',Path(__file__),R/'core/rough_adjacency.py',R/'core/gnn.py']
    meta={'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_hashes':{str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'numpy':np.__version__,'torch':torch.__version__}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    checks=edge_cases();summaries=[]
    for name in p['datasets']:
        for seed in p['seeds']:summaries.append(run(name,seed,p,out))
    meta.update(completed=True,edge_cases=checks,summaries=summaries,finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
