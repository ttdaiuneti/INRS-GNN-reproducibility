"""Locked, additive full-pipeline control study; test labels excluded from fitting."""
from pathlib import Path
import sys, json, time, hashlib, datetime, argparse, resource
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
H=Path(__file__).resolve().parent
R=H.parent
sys.path[:0]=[str(R),str(R/'shared')]
from core.data import load_planetoid_pyg
from core.gnn import GCN, set_seed
from experiments.label_safe_stream import normalized, batch_radius, rebuild

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ah(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(p)

def fit(x,a,labels,tr,va,nc,seed,cfg,p):
    # Fixed features and adjacency permit exact AX caching before lin1.
    start=time.perf_counter();set_seed(seed)
    model=GCN(x.shape[1],p['hidden'],nc,p['dropout'])
    ax=torch.sparse.mm(a,x)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
    best=-1;state=None;epoch=-1
    def forward():
        h=F.dropout(F.relu(model.lin1(ax)),p=model.dropout,training=model.training)
        return torch.sparse.mm(a,model.lin2(h))
    for e in range(p['epochs']):
        model.train();opt.zero_grad();z=forward()
        F.cross_entropy(z[tr],labels[tr]).backward();opt.step()
        model.eval()
        with torch.no_grad():
            z=forward();acc=float((z[va].argmax(1)==labels[va]).float().mean())
        if acc>best:
            best=acc;epoch=e+1;state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state);model.eval()
    with torch.no_grad():z=forward()
    return z,{'val_acc':best,'epoch':epoch,'fit_s':time.perf_counter()-start,**cfg}

def operators(x,a,assigned,trusted,seed):
    delta=batch_radius(x,assigned,trusted)
    w=rebuild(x,assigned,delta,a);w.eliminate_zeros()
    e=sp.triu(a,k=1).tocoo();i,j=e.row,e.col
    d=np.linalg.norm(x[i]-x[j],axis=1)
    scale=float(np.median(d[d>0])) if np.any(d>0) else 1.0
    f=np.exp(-d/scale)
    nw=np.asarray(w[i,j]).ravel();support=nw>0
    fs=f*support
    if fs.sum()>0:fs*=nw.sum()/fs.sum()
    rw=np.zeros(len(i));rng=np.random.default_rng(30000+seed)
    rw[rng.permutation(len(i))[:int(support.sum())]]=rng.permutation(nw[support])
    def matrix(v):
        z=sp.csr_matrix((np.r_[v,v],(np.r_[i,j],np.r_[j,i])),shape=a.shape)
        z.eliminate_zeros();return z
    assert np.array_equal(np.sort(rw[rw>0]),np.sort(nw[nw>0]))
    assert np.array_equal(fs>0,support)
    assert np.isclose(fs.sum(),nw.sum())
    return {'plain':sp.csr_matrix(a.shape),'nrs':w,'feature_all':matrix(f),'feature_support':matrix(fs),'random_support':matrix(rw)}, {'trusted_count':int(trusted.sum()),'rough_edges':int(support.sum()),'structural_edges':len(i),'feature_scale':scale,'assigned_hash':ah(assigned),'trusted_hash':ah(trusted),'nrs_values_hash':ah(nw)}

def checks(p):
    torch.set_num_threads(1);set_seed(99)
    a=normalized(sp.csr_matrix(np.array([[0,1,0],[1,0,1],[0,1,0]])))
    x=torch.randn(3,7);m=GCN(7,p['hidden'],2,p['dropout']);m.eval()
    z=m(x,a);zc=torch.sparse.mm(a,m.lin2(F.relu(m.lin1(torch.sparse.mm(a,x)))))
    assert torch.allclose(z,zc,atol=1e-7,rtol=1e-6)
    m.train();torch.manual_seed(1);z=m(x,a);z.sum().backward();g=[v.grad.clone() for v in m.parameters()]
    m.zero_grad();torch.manual_seed(1)
    h=F.dropout(F.relu(m.lin1(torch.sparse.mm(a,x))),p=m.dropout,training=True)
    torch.sparse.mm(a,m.lin2(h)).sum().backward()
    assert all(torch.allclose(v.grad,gg,atol=1e-7,rtol=1e-6) for v,gg in zip(m.parameters(),g))
    for trusted in [np.zeros(3,bool),np.ones(3,bool)]:
        operators(x.numpy().astype(float),sp.csr_matrix(np.ones((3,3))-np.eye(3)),np.array([0,1,0]),trusted,1)
    # Safe target construction is invariant to arbitrary held-out label changes.
    y=np.array([0,1,0]);tr=np.array([True,False,False]);va=np.array([False,True,False]);y2=y.copy();y2[~(tr|va)]=99
    assert np.array_equal(np.where(tr|va,y,0),np.where(tr|va,y2,0))
    return {'cached_forward_and_gradient':True,'empty_and_nonempty_control_invariants':True,'heldout_target_masking':True}

def run(name,seed,p,out):
    start=time.perf_counter()
    X,y,aa,tr,va,te,nc=load_planetoid_pyg(name)
    X=np.asarray(X,dtype=np.float64);a=sp.csr_matrix(aa,dtype=float);del aa
    assert not (tr&va).any() and not (tr&te).any() and not (va&te).any()
    x=torch.tensor(X,dtype=torch.float32)
    safe=torch.tensor(np.where(tr|va,y,0),dtype=torch.long)
    tm,vm=torch.tensor(tr),torch.tensor(va)
    z,warm=fit(x,normalized(a),safe,tm,vm,nc,10000+seed,p['warmup'],p)
    conf,pred=z.softmax(1).max(1);assigned=pred.numpy().copy();trusted=conf.numpy()>=p['tau']
    assigned[tr]=y[tr];trusted[tr]=True
    t=time.perf_counter();ops,diag=operators(X,a,assigned,trusted,seed);build=time.perf_counter()-t
    rows=[]
    for method in p['methods']:
        t=time.perf_counter();an=normalized(a+p['alpha']*ops[method]);normtime=time.perf_counter()-t
        candidates=[];best=None;bestz=None
        for k,cfg in enumerate(p['configs']):
            zz,record=fit(x,an,safe,tm,vm,nc,20000+seed,cfg,p)
            record['config_id']=k;candidates.append(record)
            if best is None or record['val_acc']>best['val_acc']:best=record;bestz=zz
        # Test targets first accessed here, after selection, never in candidates.
        test=float(np.mean(bestz.argmax(1).numpy()[te]==y[te]))
        rows.append({'method':method,'test_acc':test,'selected':best,'candidates':candidates,'normalization_s':normtime,'search_s':sum(c['fit_s'] for c in candidates)})
    result={'dataset':name,'seed':seed,'warmup':warm,'operator_build_s':build,'diagnostics':diag,'data_hashes':{'X':ah(X),'y':ah(y),'train':ah(tr),'val':ah(va),'test':ah(te),'a_indptr':ah(a.indptr),'a_indices':ah(a.indices)},'methods':rows,'elapsed_s':time.perf_counter()-start,'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write(out/f'{name}_{seed}.json',result)
    print(name,seed,{r['method']:r['test_acc'] for r in rows},'seconds',round(result['elapsed_s'],1),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check-only',action='store_true');args=ap.parse_args()
    p=json.loads((H/'protocol.json').read_text());torch.set_num_threads(1)
    tests=checks(p)
    if args.check_only:print(tests);return
    out=H/'results';out.mkdir(exist_ok=True)
    files=[H/'protocol.json',Path(__file__),R/'core/gnn.py',R/'experiments/label_safe_stream/run.py',R/'shared/graph_data.py',R/'core/rough_adjacency.py']
    hashes={str(f.relative_to(R)):sha(f) for f in files}
    meta={'started':stamp(),'source_hashes':hashes,'checks':tests,'torch':torch.__version__,'numpy':np.__version__,'completed':False}
    mp=out/'metadata.json'
    if mp.exists():
        meta=json.loads(mp.read_text());assert meta['source_hashes']==hashes,'Source changed; use separate output directory.'
    else:write(mp,meta)
    for name in p['datasets']:
        for seed in p['seeds']:
            if not (out/f'{name}_{seed}.json').exists():run(name,seed,p,out)
    meta.update(completed=True,finished=stamp());write(mp,meta)
if __name__=='__main__':main()
