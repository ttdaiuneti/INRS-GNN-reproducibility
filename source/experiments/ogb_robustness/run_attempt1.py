"""Serial multi-seed replay of the existing stateful OGB benchmark."""
import sys,json,subprocess,hashlib,datetime,os,platform
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[1]
P=json.loads((H/'protocol.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if len(sys.argv)>1:
 seed=int(sys.argv[1]);assert seed in P['seeds']
 sys.path.insert(0,str(R))
 from experiments import e2e_ogb_stream as b
 b.SEED=seed;b.K_INSERTS=P['insertions'];b.CHECKPOINTS=tuple(P['checkpoints']);b.OUT=str(H/'results'/f'seed_{seed}.json')
 assert b.INIT_FRAC==P['initial_fraction'] and b.BLOCK==P['block'] and b.CHECK_TOL==P['tolerance']
 assert not Path(b.OUT).exists()
 b.main()
else:
 out=H/'results';out.mkdir(exist_ok=False)
 files=[H/'protocol.json',Path(__file__),R/'experiments/e2e_ogb_stream.py',R/'shared/theory/incremental_nrs.py',R/'theory/incremental_rough_adjacency.py']
 inputs=[R/'shared/data/ogb'/f'arxiv_{n}.npy' for n in ['X','y','edge_index']]
 meta={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_hashes':{str(p.relative_to(R)):digest(p) for p in files},'input_hashes':{str(p.relative_to(R)):digest(p) for p in inputs},'python':sys.version,'platform':platform.platform(),'thread_environment':{k:os.getenv(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},'completed_seeds':[]}
 (out/'metadata.json').write_text(json.dumps(meta,indent=2))
 for seed in P['seeds']:
  print('Starting seed',seed,flush=True)
  with (out/f'seed_{seed}.log').open('w') as log:
   result=subprocess.run([sys.executable,str(Path(__file__)),str(seed)],stdout=log,stderr=subprocess.STDOUT)
  assert result.returncode==0,f'Seed {seed} failed: see retained log'
  meta['completed_seeds'].append(seed);(out/'metadata.json').write_text(json.dumps(meta,indent=2))
  print('Completed seed',seed,flush=True)
 meta['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();meta['completed']=True
 meta['result_hashes']={p.name:digest(p) for p in out.glob('seed_*.json')}
 (out/'metadata.json').write_text(json.dumps(meta,indent=2))
