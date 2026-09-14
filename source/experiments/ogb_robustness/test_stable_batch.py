from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'shared'))
from theory.incremental_nrs import calc_deltas_rowwise
from stable_batch import calc_deltas_blocked_stable
rng=np.random.default_rng(19)
for scale in [1,100,1e4]:
 x=rng.normal(size=(80,128))*scale;y=np.arange(80)%3
 # Opposite-class exact duplicates and nearly equal vectors; two enemies compete.
 x[1]=x[0];x[4]=x[3]+1e-9;x[5]=x[3]+2e-9
 ref=calc_deltas_rowwise(x,y)
 for block in [1,7,64]:
  actual=calc_deltas_blocked_stable(x,y,block)
  assert np.allclose(actual,ref,rtol=1e-12,atol=1e-12),(scale,block,np.max(abs(actual-ref)))
  assert actual[0]==actual[1]==0
x=rng.normal(size=(5,3));assert np.isinf(calc_deltas_blocked_stable(x,np.zeros(5))).all()
assert calc_deltas_blocked_stable(np.empty((0,3)),np.array([])).size==0
print('PASS: direct-reference tests for duplicates, near duplicates, scales, block sizes and no enemies')
