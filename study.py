"""One corrected shared-decoder study manager; never routes to the old d3 learner."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,importlib.metadata,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
from qecrobust.case_bank import get_case,names
from qecrobust.experiments import PRESETS,SUITES,ALIASES,DISTANCES,preset,suite_definition,default_seeds,default_cases
METHODS={'erm','hc','tail','rare_tail','euclid','paper_ma','legacy_ma','local_ma','multiscale_ma'}

def scientific_hash():
 files=[ROOT/'study.py',ROOT/'tables.py']
 for folder in ['decoder','scripts','qecrobust']:
  files+=sorted((ROOT/folder).glob('*.py'))
 return hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()

def environment():
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 env['PYTHONPATH']=os.pathsep.join([str(ROOT),str(ROOT/'scripts')]+([os.environ['PYTHONPATH']] if os.environ.get('PYTHONPATH') else []));return env

def split_ints(value):
 result=[int(x) for x in value.split(',') if x.strip()]
 if not result or len(result)!=len(set(result)):raise ValueError('provide a nonempty list without duplicates')
 return result

def specification(a):
 canonical,(family,distances,policy)=suite_definition(a.suite);c=preset(a.preset);preset_seed=c.pop('seed')
 # Explicit study options are overrides, never silently part of a historical reproduction.
 overrides={}
 for key in c:
  value=getattr(a,key,None)
  if value is not None:overrides[key]=value;c[key]=value
 if a.family:family=a.family
 cases=default_cases(family) if not a.cases else [x.strip() for x in a.cases.split(',') if x.strip()]
 if not cases or len(cases)!=len(set(cases)):raise ValueError('case list must be nonempty and unique')
 for case in cases:get_case(case)
 distances=split_ints(a.distances) if a.distances else list(distances)
 if a.preset.startswith('v96-') and any(d not in [15,19] for d in distances):raise ValueError('large-distance candidates require explicit --distances 15,19 (or a subset); use v9-final at 3,5,7')
 if a.preset in ('v99-d5-short','v10-d5-short') and distances!=[5]:raise ValueError('v99-d5-short is a d5-only setting; use v98-long-average for other distances')
 if any(d not in DISTANCES for d in distances):raise ValueError('supported distances are 3,5,7,15,19; d101 is not validated')
 seeds=split_ints(a.seeds) if a.seeds else default_seeds(policy,{'seed':preset_seed})
 if canonical=='corrected_preflight':
  if not a.cases:cases=[default_cases('hv')[0],default_cases('persistent')[0]]
  c.update(ntrain=50,nval=50,ntest=20,epochs=1,calibration_per_env=100)
  seeds=[107] if not a.seeds else seeds
 config=dict(c,profile=a.profile)
 codehash=scientific_hash()
 config_id=hashlib.sha256(json.dumps(dict(config=config,source=codehash),sort_keys=True).encode()).hexdigest()[:12]
 grid_id=hashlib.sha256(json.dumps([cases,distances,seeds]).encode()).hexdigest()[:12]
 base=ROOT/'results'/canonical/a.preset/a.profile/config_id
 jobs=[]
 for distance in distances:
  for seed in seeds:
   for case in cases:
    out=base/f'd{distance}'/f'{case}_seed{seed}.json'
    cmd=[sys.executable,'decoder/run_study.py','--case',case,'--distance',str(distance),'--seed',str(seed),'--out',str(out.relative_to(ROOT))]
    for key,value in config.items():
     if isinstance(value,bool):
      if value:cmd.append('--'+key.replace('_','-'))
     else:cmd+=['--'+key.replace('_','-'),str(value)]
    signature=hashlib.sha256((json.dumps(cmd)+codehash).encode()).hexdigest()
    jobs.append(dict(out=str(out.relative_to(ROOT)),command=cmd,signature=signature,case=case,distance=distance,seed=seed))
 return dict(schema=1,implementation='corrected_shared_gru',version='10.0',suite=canonical,preset=a.preset,profile=a.profile,configuration=config,overrides=overrides,preset_seed=preset_seed,source_hash=codehash,config_id=config_id,grid_id=grid_id,cases=cases,distances=distances,seeds=seeds,total=len(jobs),base=str(base.relative_to(ROOT)),jobs=jobs)

def complete(job):
 out=ROOT/job['out'];receipt=out.with_suffix('.receipt.json')
 try:
  m=json.loads(receipt.read_text());r=json.loads(out.read_text())
  return m['signature']==job['signature'] and m['exit_code']==0 and set(r['methods'])==METHODS and r.get('implementation')=='corrected_shared_gru'
 except (OSError,ValueError,KeyError):return False

def run(a):
 manifest=specification(a);done=sum(complete(j) for j in manifest['jobs'])
 print(json.dumps({k:v for k,v in manifest.items() if k!='jobs'},indent=2),flush=True)
 print(f"total={manifest['total']} done={done} pending={manifest['total']-done}; corrected shared GRU; CPU",flush=True)
 if a.command in ['plan','status']:return
 if a.gpus not in ['none','cpu','auto']:raise ValueError('this corrected implementation is CPU-only; use --gpus none')
 if a.threads_per_task!=1:raise ValueError('reproducible CPU profile uses one thread per task')
 base=ROOT/manifest['base'];folder=base/'manifests';folder.mkdir(parents=True,exist_ok=True)
 mp=folder/(manifest['grid_id']+'.json');mp.write_text(json.dumps(manifest,indent=2))
 (base/'LATEST.json').write_text(json.dumps({'manifest':str(mp.relative_to(ROOT))}))
 workers=min(4,os.cpu_count() or 1) if a.workers=='auto' else int(a.workers)
 if workers<1 or a.timeout_seconds<=0:raise ValueError('workers and timeout must be positive')
 def one(job):
  if complete(job):return 0
  out=ROOT/job['out'];out.parent.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
  with out.with_suffix('.log').open('w') as f:
   try:rc=subprocess.run(job['command'],cwd=ROOT,env=environment(),stdout=f,stderr=subprocess.STDOUT,timeout=a.timeout_seconds).returncode
   except subprocess.TimeoutExpired:f.write('Task timed out. Output is not accepted.\n');rc=124
  if rc==0:
   try:
    result=json.loads(out.read_text())
    if set(result['methods'])!=METHODS or result['implementation']!='corrected_shared_gru':raise ValueError('wrong backend or missing comparator')
    result['experiment_provenance']={k:manifest[k] for k in ['preset','profile','configuration','source_hash','grid_id']}
    out.write_text(json.dumps(result,indent=2))
   except (OSError,ValueError,KeyError) as error:
    rc=1
    with out.with_suffix('.log').open('a') as f:f.write('\nResult validation failed: '+str(error)+'\n')
  out.with_suffix('.receipt.json').write_text(json.dumps(dict(signature=job['signature'],exit_code=rc,seconds=time.perf_counter()-start,command=job['command']),indent=2))
  print(f"{'OK' if rc==0 else 'FAIL'} rc={rc} {job['out']}",flush=True)
  if rc:print('\n'.join(out.with_suffix('.log').read_text(errors='replace').splitlines()[-12:]),flush=True)
  return rc
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:codes=list(pool.map(one,manifest['jobs']))
 print(f'finished: ok={sum(c==0 for c in codes)} failed={sum(c!=0 for c in codes)}',flush=True)
 if any(codes):raise SystemExit(2)

def doctor():
 missing=[]
 for module in ['torch','numpy','scipy','pymatching','ldpc','panqec']:
  try:print(module,importlib.metadata.version(module))
  except importlib.metadata.PackageNotFoundError:missing.append(module);print(module,'MISSING')
 print('This command inspects package metadata only. Run the listed validation commands to test binary imports/physics.')
 if missing:raise SystemExit(2)

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 sub.add_parser('list-suites');l=sub.add_parser('list-cases');l.add_argument('--family',choices=['hv','hv_band','band','persistent','candidate','noise-geometry','weights','correlation-profiles','predictive-average','output-policy','segment-aliases','v10-policy','all'],default='all')
 sub.add_parser('list-presets');sub.add_parser('doctor')
 for command in ['run','plan','status']:
  q=sub.add_parser(command);q.add_argument('--suite',required=True);q.add_argument('--preset',choices=list(PRESETS),default='v9-final');q.add_argument('--profile',choices=['matched','original'],default='matched')
  q.add_argument('--family',choices=['hv','hv_band','band','persistent','candidate','noise-geometry','weights','correlation-profiles','predictive-average','output-policy','all']);q.add_argument('--cases');q.add_argument('--distances');q.add_argument('--seeds')
  q.add_argument('--workers',default='4');q.add_argument('--gpus',default='none');q.add_argument('--threads-per-task',type=int,default=1);q.add_argument('--timeout-seconds',type=int,default=1800)
  for key in ['ntrain','nval','ntest','epochs','calibration_per_env']:q.add_argument('--'+key.replace('_','-'),type=int)
  for key in ['rho','hazard_kappa','strength','train_stay','test_stay','deployment_source_mass']:q.add_argument('--'+key.replace('_','-'),type=float)
  q.add_argument('--history',type=int,choices=[8,14]);q.add_argument('--evaluate-control',action='store_true',default=None)
  for key,choices in [('witness_sampling',['nominal','balanced']),('checkpoint_mode',['last','late_average']),('evaluation_role',['development','test']),('weight_normalization',['none','self_normalized']),('noise_geometry',['standard','ribbons','segments','long_segments','long_aligned','distance_scaled']),('regret_reference',['environment','selector','soft_selector']),('auxiliary_target',['environment','correct_expert']),('feature_mode',['legacy','action_history']),('cost_units',['raw','bank','pre_hazard_bank']),('adversary_search',['local','rays']),('adversary_loss',['nll','logical_error']),('expert_calibration',['unconditional','action']),('witness_mode',['disagreement','coset']),('optimizer_mode',['iterative','one_shot']),('hc_cost_normalization',['hazard','unit_median'])]:q.add_argument('--'+key.replace('_','-'),choices=choices)
 t=sub.add_parser('tables',aliases=['paper-tables']);t.add_argument('--manifest');t.add_argument('--suite');t.add_argument('--preset',choices=list(PRESETS),default='v9-final');t.add_argument('--profile',choices=['matched','original'],default='matched');t.add_argument('--allow-partial',action='store_true')
 c=sub.add_parser('check');c.add_argument('--group',choices=['bank','physics','actions','corr-min','cosets','optimizer','persistent','candidate','noise-geometry','weights','correlation-profiles','predictive-average','output-policy','segment-aliases','v10-policy','all'],default='all')
 a=p.parse_args()
 if a.command=='list-suites':
  print('suite | cases | distances | default seeds | total tasks')
  for name,(family,distances,policy) in SUITES.items():
   nc=2 if name=='corrected_preflight' else len(default_cases(family));ns=5 if policy=='five' else 3 if policy=='three' else 1
   print(f'{name} | {nc} | {len(distances)} | {ns} | {nc*len(distances)*ns}')
  print('Aliases:',json.dumps(ALIASES,sort_keys=True));return
 if a.command=='list-cases':
  for name in names(a.family):
   case=get_case(name);print(name,'family='+case['family'],'modeled=5','excluded_control='+str(case['control'] is not None))
  return
 if a.command=='list-presets':print(json.dumps(PRESETS,indent=2));return
 if a.command=='doctor':return doctor()
 if a.command in ['run','plan','status']:return run(a)
 if a.command in ['tables','paper-tables']:
  from tables import write_tables
  return write_tables(a)
 if a.command=='check':
  files={'bank':'checks/bank.py','physics':'decoder/tests.py','actions':'decoder/validate_action.py','corr-min':'decoder/validate_corr_min.py','cosets':'decoder/validate_cosets.py','optimizer':'decoder/test_ray_adversary.py','persistent':'checks/persistent.py','candidate':'checks/large_candidate.py','noise-geometry':'checks/noise_geometry.py','weights':'checks/importance_weights.py','correlation-profiles':'checks/correlation_profiles.py','predictive-average':'checks/predictive_average.py','output-policy':'checks/output_policy.py','segment-aliases':'checks/segment_aliases.py','v10-policy':'checks/v10_policy.py'}
  for key in files if a.group=='all' else [a.group]:
   cp=subprocess.run([sys.executable,files[key]],cwd=ROOT,env=environment())
   if cp.returncode:raise SystemExit(cp.returncode)
if __name__=='__main__':main()
