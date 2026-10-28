"""Independent small-case checks for labels, radius/weights and training cache."""
from pathlib import Path
import json, sys
import numpy as np
import scipy.sparse as sp
import torch
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H))
from run import fit, operators, normalized, checks
p=json.loads((H/'protocol.json').read_text());checks(p)
torch.set_num_threads(1)
rng=np.random.default_rng(555)
x=rng.normal(size=(9,5));x[1]=x[0]
a=sp.csr_matrix(np.ones((9,9))-np.eye(9))
labels=np.array([0,1,0,1,0,1,0,1,0]);trusted=np.array([1,1,1,0,1,0,1,0,0],bool)
ops,_=operators(x,a,labels,trusted,0)
# Scalar oracle independent of the reused batch radius/weight functions.
d=np.full(9,np.inf)
for i in range(9):
    if trusted[i]:
        ds=[np.sqrt(sum((x[i,k]-x[j,k])**2 for k in range(5))) for j in range(9) if trusted[j] and labels[i]!=labels[j]]
        if ds:d[i]=min(ds)
w=np.zeros((9,9))
for i in range(9):
    for j in range(9):
        if i!=j and np.isfinite(d[i]) and np.isfinite(d[j]):
            w[i,j]=np.exp(-np.linalg.norm(x[i]-x[j])/(d[i]+d[j]+1e-9))
assert np.allclose(w,ops['nrs'].toarray(),atol=1e-14,rtol=1e-13)
tr=np.array([1,1,0,0,0,0,0,0,0],bool);va=np.array([0,0,1,1,0,0,0,0,0],bool)
y2=labels.copy();y2[~(tr|va)]=1-y2[~(tr|va)]
pp=dict(p,epochs=8)
z=[];struct=[]
for y in [labels,y2]:
    zz,_=fit(torch.tensor(x,dtype=torch.float32),normalized(a),torch.tensor(np.where(tr|va,y,0)),torch.tensor(tr),torch.tensor(va),2,111,p['warmup'],pp)
    conf,pred=zz.softmax(1).max(1);assigned=pred.numpy().copy();assigned[tr]=y[tr]
    trust=conf.numpy()>=p['tau'];trust[tr]=True
    op,_=operators(x,a,assigned,trust,7)
    z.append(zz);struct.append(op)
assert torch.equal(z[0],z[1])
for method in p['methods']:assert np.array_equal(struct[0][method].toarray(),struct[1][method].toarray())
report={'scalar_radius_weight_reference':True,'heldout_label_scramble_warmup_logits_exact':True,'heldout_label_scramble_all_operators_exact':True,**checks(p)}
(H/'validation.json').write_text(json.dumps(report,indent=2));print(report)
