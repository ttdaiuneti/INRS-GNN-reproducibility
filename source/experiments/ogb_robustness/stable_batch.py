"""Blocked Gram nearest-enemy radii with direct refinement near cancellation.

Ordinary-distance candidates use the original Gram computation. For rows whose
minimum squared enemy distance is within a conservative roundoff threshold,
recompute all enemy candidates in that near-minimum band by direct subtraction.
This is a numerical guard, not a change to the NRS radius definition.
"""
import numpy as np

def calc_deltas_blocked_stable(X,y,block=2048):
 X=np.ascontiguousarray(X,dtype=np.float64);y=np.asarray(y);n=len(y)
 if n==0:return np.array([])
 sq=np.einsum('ij,ij->i',X,X);out=np.full(n,np.inf)
 # Conservative scale, including dimension, to catch nearly identical vectors.
 guard=32*max(1,X.shape[1])*np.finfo(np.float64).eps*(sq+sq.max())
 for start in range(0,n,block):
  end=min(start+block,n)
  with np.errstate(divide='ignore',over='ignore',invalid='ignore'):g=X[start:end]@X.T
  d2=sq[start:end,None]+sq[None,:]-2*g
  np.maximum(d2,0,out=d2);d2[y[start:end,None]==y[None,:]]=np.inf
  minima=d2.min(axis=1)
  values=np.sqrt(minima)
  for local in np.flatnonzero(minima<=guard[start:end]):
   row=start+local;candidates=np.flatnonzero(d2[local]<=minima[local]+2*guard[row])
   diff=X[candidates]-X[row]
   values[local]=np.sqrt(np.sum(diff*diff,axis=1)).min()
  out[start:end]=values
 return out
