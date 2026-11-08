"""Manifest-checked tables for the corrected shared learner only."""
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parent
METHODS=['erm','hc','tail','rare_tail','euclid','paper_ma','legacy_ma','local_ma','multiscale_ma']
MAIN=['erm','hc','tail','euclid','paper_ma','legacy_ma']
REFS=['mwpm','corr_min','source_hmm','bayes']

def csvwrite(path,rows):
 if not rows:return
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def format_values(values):
 values=[x for x in values if x is not None]
 if not values:return '—'
 return f'{100*np.mean(values):.3f}%' if len(values)==1 else f'{100*np.mean(values):.3f} ± {100*np.std(values,ddof=1):.3f}%'

def write_tables(a):
 from study import scientific_hash
 from qecrobust.experiments import suite_definition
 code=scientific_hash()
 if a.manifest:
  mp=Path(a.manifest);paths=[mp if mp.is_absolute() else ROOT/mp]
 else:
  if not a.suite:raise ValueError('specify --suite or --manifest')
  canonical,_=suite_definition(a.suite);paths=[]
  for latest in (ROOT/'results'/canonical/a.preset/a.profile).glob('*/LATEST.json'):
   paths.append(ROOT/json.loads(latest.read_text())['manifest'])
 if not paths:raise FileNotFoundError('no launched study found; run the study first')
 generated=0
 for path in sorted(paths):
  m=json.loads(path.read_text())
  if m['source_hash']!=code:
   if a.manifest:raise ValueError('manifest uses different source code')
   print('Skipping stale source configuration:',path.relative_to(ROOT));continue
  rows=[];missing=[]
  for job in m['jobs']:
   p=ROOT/job['out'];receipt=p.with_suffix('.receipt.json')
   try:
    r=json.loads(p.read_text());meta=json.loads(receipt.read_text())
    if meta['exit_code']!=0 or meta['signature']!=job['signature']:raise ValueError('failed/stale receipt')
    if r['implementation']!='corrected_shared_gru' or set(r['methods'])!=set(METHODS):raise ValueError('wrong backend/missing methods')
    if (r['suite'],r['distance'],r['seed'])!=(job['case'],job['distance'],job['seed']):raise ValueError('record identity mismatch')
    for v in r['methods'].values():
     if not 0<=v['ler']<=1:raise ValueError('invalid logical error rate')
    if r['distance']>3 and r['references']['bayes'] is not None:raise ValueError('approximate result mislabeled Bayes')
    if r['case_family']=='persistent' and (r['modeled_support']!=[0,1,2,3,4] or r['mass_on_diagnostic_control']!=0 or not r['diagnostic_control_excluded']):raise ValueError('diagnostic control entered modeled support')
    rows.append(r)
   except (OSError,KeyError,ValueError) as error:missing.append((job['out'],str(error)))
  if missing and not a.allow_partial:raise ValueError(f'{len(missing)} missing/failed/stale tasks; first: {missing[0]}')
  if not rows:raise ValueError('no valid records to tabulate')
  out=ROOT/m['base']/'tables'/m['grid_id'];out.mkdir(parents=True,exist_ok=True)
  title=['# Corrected shared-decoder study', '', 'Status: '+('PARTIAL DEBUG OUTPUT' if missing else 'COMPLETE'),'',f"Suite: {m['suite']}; preset: {m['preset']}; profile: {m['profile']}; tasks: {len(rows)}/{m['total']}",'', 'Single seed: LER. Multiple seeds: mean ± sample SD across model/test seeds. Calibration is fixed.','', '| Case | Space | d | Seeds | ERM | HC-MA† | Tail | Euclid | Paper MA† | Legacy MA† | MWPM | Corr-Min† | Source-HMM† | Bayes |','|'+'---|'*14]
  title.insert(6,'Witness sampling: '+str(m['configuration'].get('witness_sampling','nominal'))+'.')
  title.insert(6,'Decoder output: '+str(m['configuration'].get('checkpoint_mode','last'))+'.')
  title.insert(6,'Evaluation role: '+str(m['configuration'].get('evaluation_role','test'))+'. Development outcomes are not confirmatory test evidence.')
  title.insert(6,'Importance estimator: '+str(m['configuration'].get('weight_normalization','none'))+'.')
  title.insert(6,'Noise geometry: '+str(m['configuration'].get('noise_geometry','standard'))+'; segment lengths scale with distance when enabled.')
  title.insert(6,'Variant: '+str(m['configuration'].get('regret_reference','environment'))+' reference; '+str(m['configuration'].get('feature_mode','legacy'))+' features; '+str(m['configuration'].get('cost_units','raw'))+' costs. Candidate presets are not established all-method improvements.')
  if m['configuration'].get('regret_reference','environment')!='environment':title=[line.replace('HC-MA†','HC-candidate†') for line in title]
  extended=['| Case | d | Rare-tail | Local-MA† | Multiscale-MA† | Target-HMM approx. | Known-env reference |','|'+'---|'*7]
  raw=[];paired=[];controls=[];distance=[]
  for d in m['distances']:
   for case in m['cases']:
    group=[r for r in rows if r['distance']==d and r['suite']==case]
    if not group:continue
    method_values=[[r['methods'][method]['ler'] for r in group] for method in MAIN]
    ref_values=[[None if r['references'][ref] is None else r['references'][ref]['ler'] for r in group] for ref in REFS]
    title.append('| '+' | '.join([case,group[0]['case_family'],str(d),str(len(group))]+[format_values(v) for v in method_values+ref_values])+' |')
    more=[[r['methods'][method]['ler'] for r in group] for method in ['rare_tail','local_ma','multiscale_ma']]
    more += [[None if r['references'][ref] is None else r['references'][ref]['ler'] for r in group] for ref in ['target_hmm_approx','known_environment']]
    extended.append('| '+' | '.join([case,str(d)]+[format_values(v) for v in more])+' |')
  for r in rows:
   for method,v in r['methods'].items():raw.append(dict(case=r['suite'],space=r['case_family'],distance=r['distance'],seed=r['seed'],method=method,ler=v['ler']))
   for ref,v in r['references'].items():raw.append(dict(case=r['suite'],space=r['case_family'],distance=r['distance'],seed=r['seed'],method=ref,ler=None if v is None else v['ler']))
   for method,v in r['paired'].items():paired.append(dict(case=r['suite'],distance=r['distance'],seed=r['seed'],method=method,erm_minus_method=v['gain'],ci95_lower=v['ci95'][0],ci95_upper=v['ci95'][1]))
   if 'diagnostic_control' in r:
    c=r['diagnostic_control']
    for method,value in c['methods'].items():controls.append(dict(case=r['suite'],distance=r['distance'],seed=r['seed'],method=method,control_ler=value,included_in_transport=False))
  for d in m['distances']:
   sub=[r for r in rows if r['distance']==d]
   if sub:
    e=float(np.mean([r['methods']['erm']['ler'] for r in sub]));h=float(np.mean([r['methods']['hc']['ler'] for r in sub]));b=float(np.mean([r['references']['mwpm']['ler'] for r in sub]))
    distance.append(dict(distance=d,cases=len({r['suite'] for r in sub}),seeds=len({r['seed'] for r in sub}),erm=e,hc=h,mwpm=b,hc_relative_reduction_vs_erm=None if not e else (e-h)/e,hc_relative_reduction_vs_mwpm=None if not b else (b-h)/b))
  title += ['', '† Above d=3, physical experts, source-HMM and logical witnesses are approximations; exact Bayes is unavailable. d=3 uses exact conditional physical tables within the shared learner, not the old standalone d3 paper learner.','', 'Paired sequence intervals condition on fitted models; they do not include model/calibration-seed uncertainty. Distance means are descriptive.']
  (out/'MAIN.md').write_text('\n'.join(title)+'\n');(out/'EXTENDED.md').write_text('\n'.join(extended)+'\n')
  lastrows=[]
  for record in rows:
   for method,value in record.get('last_iterate_controls',{}).items():lastrows.append(dict(case=record['suite'],distance=record['distance'],seed=record['seed'],method=method,ler=value['ler'],errors=value['errors'],n=value['n']))
  csvwrite(out/'LAST_ITERATE_CONTROLS.csv',lastrows)
  csvwrite(out/'ALL_METHODS.csv',raw);csvwrite(out/'PAIRED.csv',paired);csvwrite(out/'DISTANCE_MEANS.csv',distance);csvwrite(out/'DIAGNOSTIC_CONTROL.csv',controls)
  print('Tables:',out.relative_to(ROOT));generated+=1
 if not generated:raise ValueError('no current-source manifest found')
