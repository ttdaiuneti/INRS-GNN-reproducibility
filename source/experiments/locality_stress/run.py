"""Reproducible adverse-locality benchmark. Run from any directory."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import sys,json,csv,time,hashlib,platform,resource,datetime
from pathlib import Path
import numpy as np
import scipy
from scipy.sparse import csr_matrix,block_diag
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'shared'));sys.path.insert(0,str(ROOT))
from theory.incremental_nrs import calc_deltas_rowwise,calc_deltas_vectorized,incremental_insert
from theory.incremental_rough_adjacency import incremental_rough_adjacency_update_sparse,batch_rough_adjacency_edgelist,nrs_psi
HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def construct(n,seed,scenario):
 rng=np.random.default_rng(seed); y=np.repeat([0,1],n//2); X=rng.normal(0,.02,(n,P['features'])); X[y==1,0]+=10
 if scenario.startswith('ring'): u=np.arange(n);v=(u+1)%n
 else:u=np.zeros(n-1,dtype=int);v=np.arange(1,n)
 a=csr_matrix((np.ones(2*len(u),dtype=bool),(np.r_[u,v],np.r_[v,u])),shape=(n,n))
 z=np.zeros(P['features']);z[0]=0 if scenario=='ring_enemy' else 30
 return X,y,a,z,1 if scenario=='ring_enemy' else 0,0 if scenario=='star_hub' else 1

def extend(a,nb):
 n=a.shape[0]; coo=a.tocoo()
 return csr_matrix((np.ones(len(coo.data)+2,dtype=bool),(np.r_[coo.row,nb,n],np.r_[coo.col,n,nb])),shape=(n+1,n+1))
def weights(X,y,d,a):
 u,v,w=batch_rough_adjacency_edgelist(X,y,d,a,np.arange(X.shape[1]))
 return {(int(i),int(j)):float(z) for i,j,z in zip(u,v,w)}|{(int(j),int(i)):float(z) for i,j,z in zip(u,v,w)}
def inc(X,y,a,d,W,z,label,nb):
 t=time.perf_counter_ns();xn,yn,dn,_,shrink=incremental_insert(X,y,d,z,label);t1=time.perf_counter_ns()
 an=extend(a,nb); touch={len(y),nb}; t2=time.perf_counter_ns()
 updates,T=incremental_rough_adjacency_update_sparse(xn,yn,dn,an,np.arange(X.shape[1]),touch,shrink)
 W.update(updates);t3=time.perf_counter_ns()
 return (xn,yn,dn,an,W,shrink,T),dict(inc_delta_ms=(t1-t)/1e6,inc_bookkeeping_ms=(t2-t1)/1e6,inc_w_ms=(t3-t2)/1e6,inc_total_ms=(t3-t)/1e6)
def batch(X,y,a,z,label,nb):
 t=time.perf_counter_ns();xn=np.vstack([X,z]);yn=np.append(y,label);an=extend(a,nb);t1=time.perf_counter_ns()
 dn=calc_deltas_vectorized(xn,yn);t2=time.perf_counter_ns();W=weights(xn,yn,dn,an);t3=time.perf_counter_ns()
 return (dn,W),dict(batch_bookkeeping_ms=(t1-t)/1e6,batch_delta_ms=(t2-t1)/1e6,batch_w_ms=(t3-t2)/1e6,batch_total_ms=(t3-t)/1e6)
def validate(state,bat,n,scenario):
 X,y,d,a,W,S,T=state; ref=calc_deltas_rowwise(X,y);assert np.array_equal(np.isinf(d),np.isinf(ref))
 finite=np.isfinite(ref);err=float(np.max(np.abs(d[finite]-ref[finite])))
 coo=a.tocoo();support=set(zip(coo.row.tolist(),coo.col.tolist()));assert set(W)==support
 wr={}
 for u,v in support:
  if u<v:
   val=nrs_psi(float(np.linalg.norm(X[u]-X[v])),ref[u],ref[v]);wr[u,v]=wr[v,u]=val
 werr=max(abs(W[e]-wr[e]) for e in support);sym=max(abs(W[u,v]-W[v,u]) for u,v in support)
 bd,bw=bat;assert set(bw)==support; assert np.array_equal(np.isinf(bd),np.isinf(ref))
 berr=float(np.max(np.abs(bd[finite]-ref[finite])));bwerr=max(abs(bw[e]-wr[e]) for e in support)
 q=int(np.diff(a.indptr)[list(T)].sum()); changed_edges=sum(1 for u,v in support if u<v and (u in T or v in T))
 assert len(S)==(n//2 if scenario=='ring_enemy' else 0),(scenario,len(S))
 assert len(T)==(n//2+1 if scenario=='ring_enemy' else 2)
 expected_q={'ring_control':4,'ring_enemy':n+2,'star_leaf':3,'star_hub':n+1}[scenario]
 assert q==expected_q,(q,expected_q)
 assert max(err,werr,sym,berr,bwerr)<=P['tolerance'],(err,werr,sym,berr,bwerr)
 return dict(shrink=len(S),touch=len(T),q_touch=q,edges_updated=changed_edges,edges_total=len(support)//2,delta_error=err,weight_error=werr,symmetry_error=sym,batch_delta_error=berr,batch_weight_error=bwerr,correctness_pass=True)
def main():
 out=HERE/'results';out.mkdir(exist_ok=False)
 meta={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':sha(HERE/'protocol.json'),'script_sha256':sha(Path(__file__)),'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':{k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'shared/theory/incremental_nrs.py',ROOT/'theory/incremental_rough_adjacency.py']}}
 (out/'metadata.json').write_text(json.dumps(meta,indent=2))
 with (out/'raw_results.csv').open('w') as f:
  writer=None
  for n in P['sizes']:
   for seed in P['seeds']:
    for scenario in P['scenarios']:
     X,y,a,z,label,nb=construct(n,seed,scenario);d=calc_deltas_rowwise(X,y);W=weights(X,y,d,a)
     inc(X,y,a,d,W.copy(),z,label,nb);batch(X,y,a,z,label,nb)
     for rep in range(P['repetitions']):
      wcopy=W.copy()
      if (rep+seed)%2==0:state,it=inc(X,y,a,d,wcopy,z,label,nb);bat,bt=batch(X,y,a,z,label,nb)
      else:bat,bt=batch(X,y,a,z,label,nb);state,it=inc(X,y,a,d,wcopy,z,label,nb)
      checks=validate(state,bat,n,scenario)
      row=dict(scenario=scenario,n_old=n,seed=seed,repeat=rep,features=P['features'],**checks,**it,**bt,input_payload_bytes=X.nbytes+y.nbytes+d.nbytes+a.data.nbytes+a.indices.nbytes+a.indptr.nbytes)
      if writer is None:writer=csv.DictWriter(f,fieldnames=row.keys());writer.writeheader()
      writer.writerow(row);f.flush()
     print(n,seed,scenario,'PASS',flush=True)
 meta['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();meta['suite_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024);meta['raw_sha256']=sha(out/'raw_results.csv');meta['completed']=True
 (out/'metadata.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
