"""Rebuild all summaries from complete, hash-verified raw seed records."""
from pathlib import Path
import json, hashlib, csv
import numpy as np
from scipy import stats
H=Path(__file__).resolve().parent;R=H.parents[1]
p=json.loads((H/'protocol.json').read_text());out=H/'results'
meta=json.loads((out/'metadata.json').read_text())
assert meta['completed']
for f,h in meta['source_hashes'].items():assert hashlib.sha256((R/f).read_bytes()).hexdigest()==h,f
raw=[];aggregates=[];comparisons=[]
for ds in p['datasets']:
    runs=[json.loads((out/f'{ds}_{s}.json').read_text()) for s in p['seeds']]
    assert all(r['dataset']==ds and r['seed']==s for r,s in zip(runs,p['seeds']))
    values={}
    for method in p['methods']:
        rows=[next(m for m in r['methods'] if m['method']==method) for r in runs]
        for r,row in zip(runs,rows):
            assert len(row['candidates'])==len(p['configs'])
            assert row['selected']==max(row['candidates'],key=lambda c:c['val_acc'])
            assert all('test_acc' not in c for c in row['candidates'])
            raw.append({'dataset':ds,'seed':r['seed'],'method':method,'test_acc':row['test_acc'],'val_acc':row['selected']['val_acc'],'config_id':row['selected']['config_id'],'epoch':row['selected']['epoch'],'search_s':row['search_s'],'warmup_s_shared':r['warmup']['fit_s'],'all_operators_build_s_shared':r['operator_build_s'],'trusted_count':r['diagnostics']['trusted_count'],'rough_edges':r['diagnostics']['rough_edges']})
        vals=np.array([r['test_acc'] for r in rows])*100;values[method]=vals
        aggregates.append({'dataset':ds,'method':method,'mean_pct':float(vals.mean()),'sample_sd_pct':float(vals.std(ddof=1)),'mean_search_s':float(np.mean([r['search_s'] for r in rows]))})
    for method in p['methods']:
        if method=='nrs':continue
        d=values['nrs']-values[method];mean=float(d.mean());se=stats.sem(d)
        ci=stats.t.interval(.95,len(d)-1,loc=mean,scale=se) if se else (mean,mean)
        pv=float(stats.ttest_1samp(d,0).pvalue) if se else (1.0 if mean==0 else 0.0)
        comparisons.append({'dataset':ds,'control':method,'difference_pp':mean,'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),'p_raw':pv,'wins':int((d>0).sum()),'ties':int((d==0).sum()),'losses':int((d<0).sum()),'paired_differences_pp':d.tolist()})
order=np.argsort([r['p_raw'] for r in comparisons]);running=0
for k,i in enumerate(order):
    running=max(running,min(1,comparisons[i]['p_raw']*(len(order)-k)));comparisons[i]['p_holm']=running
passed=all(r['difference_pp']>0 for r in comparisons) and all(r['p_holm']<.05 for r in comparisons if r['control']=='plain')
summary={'aggregates':aggregates,'comparisons':comparisons,'temporal_gate_passed':passed,'seed_records':len(p['datasets'])*len(p['seeds']),'downstream_fits':len(raw)*len(p['configs']),'warmup_fits':len(p['datasets'])*len(p['seeds']),'total_seed_elapsed_s':sum(json.loads(f.read_text())['elapsed_s'] for f in out.glob('*_*.json')),'raw_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.glob('*_*.json'))}}
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
with (out/'selected_runs.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
lines=['# Full-pipeline controls — 2026-09-13','','Protocol locked before these new runs. This is a bounded control study on two fixed public splits; it is not evidence over split/dataset populations. Sample SD uses ddof=1.','',f"Completed: {summary['warmup_fits']} fresh warmups + {summary['downstream_fits']} downstream fits (200 epochs each). Alpha fixed to 1. Four validation-selected learning-rate/weight-decay configurations per method; no test selection.",'','| Dataset | Method | Accuracy %, mean ± sample SD | Mean downstream search s |','|---|---|---:|---:|']
for r in aggregates:lines.append(f"| {r['dataset']} | {r['method']} | {r['mean_pct']:.2f} ± {r['sample_sd_pct']:.2f} | {r['mean_search_s']:.2f} |")
lines+=['','| Dataset | NRS minus control | Difference pp | Unadjusted paired 95% CI | Holm p (8 tests) | W/T/L |','|---|---|---:|---|---:|---|']
for r in comparisons:lines.append(f"| {r['dataset']} | {r['control']} | {r['difference_pp']:+.2f} | [{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}] | {r['p_holm']:.4f} | {r['wins']}/{r['ties']}/{r['losses']} |")
lines+=['',f"Temporal utility expansion gate: **{'PASS' if passed else 'NO-GO'}**.",'','The gate is an internal rule for allocating further experimental effort, not a statistical definition of Q1 quality. No-go does not establish absence of all NRS utility. Do not reuse the old fixed-graph TOST conclusion as a full-pipeline equivalence result. CIs are not multiplicity-adjusted.','', 'Controls: feature_all uses an unsupervised distance kernel on all structural edges. feature_support uses that kernel on NRS support and matches total rough weight. random_support relocates the exact positive NRS weight multiset to random structural edges; node degrees are not preserved. It is not a same-support weight shuffle. Both support controls depend on NRS preprocessing.','', 'Timing: downstream search columns exclude warmup and operator construction; raw CSV/JSON include those shared costs separately. Build time combines all operators and cannot be attributed to NRS alone. Peak RSS is cumulative process high-water mark, not per-method memory. These are snapshot CPU costs, not an incremental speedup benchmark.','', 'Limitations: two fixed public splits; fixed tau and alpha; four candidate configurations; no temporal prediction, multiple splits, or optimized nearest-enemy comparator. Feature-support control isolates kernel weighting conditional on NRS support, not superiority of NRS support over all alternatives.']
(R/'docs/full_pipeline_controls_results_2026-09-13.md').write_text('\n'.join(lines)+'\n')
# Short publication table: accuracy and paired comparisons against plain.
tex=[r'\begin{table}[htbp]',r'\centering',r'\caption{Full-pipeline control study: ten fresh warm-up and downstream seeds per dataset, with four downstream configurations per method and fixed $\alpha=1$. Accuracy is mean $\pm$ sample SD (\%). These runs are separate from the fixed-graph equivalence study.}',r'\label{tab:full-pipeline}',r'\begin{tabular}{lrr}',r'\toprule',r'Method & Cora & CiteSeer \\',r'\midrule']
names={'plain':'Plain GCN','nrs':'NRS hybrid','feature_all':'Feature distance, all edges','feature_support':'Feature distance, NRS support','random_support':'Random support, NRS weight multiset'}
for method in p['methods']:
    cols=[next(a for a in aggregates if a['dataset']==ds and a['method']==method) for ds in p['datasets']]
    tex.append(names[method]+' & '+' & '.join(f"${r['mean_pct']:.2f} \\pm {r['sample_sd_pct']:.2f}$" for r in cols)+r' \\')
tex += [r'\midrule']
comp=[next(c for c in comparisons if c['dataset']==ds and c['control']=='plain') for ds in p['datasets']]
tex.append('NRS minus plain (pp)'+' & '+' & '.join(f"${r['difference_pp']:+.2f}$" for r in comp)+r' \\')
tex.append('Paired 95\\% CI (pp)'+' & '+' & '.join(f"$[{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}]$" for r in comp)+r' \\')
tex.append('Holm-adjusted $p$'+' & '+' & '.join(f"${r['p_holm']:.3f}$" for r in comp)+r' \\')
tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(H/'tab_full_pipeline.tex').write_text('\n'.join(tex)+'\n')
print(json.dumps(summary,indent=2))
