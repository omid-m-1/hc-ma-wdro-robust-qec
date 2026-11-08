from pathlib import Path
import os,sys,argparse,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(Path(__file__).parent),str(ROOT),str(ROOT/'scripts')]
import numpy as np,torch
from physics import setup,Experts,draw
from geometry import costs
from learning import METHODS,fit,prediction,action_history
from qecrobust.case_bank import get_case,names


def run(case,distance,profile='matched',strength=.6,seed=107,ntest=500,ntrain=300,nval=100,epochs=24,rho=.5,optimizer_mode='iterative',methods=None,deployment_source_mass=0.,expert_calibration='action',adversary_loss='logical_error',witness_mode='coset',hc_cost_normalization='hazard',adversary_search='local',calibration_per_env=600,history=8,train_stay=.95,test_stay=.98,hazard_kappa=3.,evaluate_control=False,regret_reference="environment",auxiliary_target="environment",feature_mode="legacy",cost_units="raw",noise_geometry="standard",weight_normalization="none",evaluation_role="test",checkpoint_mode="last",witness_sampling="nominal"):
    if any(int(x)<=0 for x in [ntest,ntrain,nval,epochs]):raise ValueError('ntrain, nval, ntest and epochs must be positive')
    if not np.isfinite(rho) or rho<0:raise ValueError('rho must be finite and nonnegative')
    if not 0<float(strength)<=1:raise ValueError('strength must lie in (0,1]')
    if history not in [8,14]:raise ValueError('history must be 8 or 14')
    if not (0<=train_stay<1 and 0<=test_stay<1):raise ValueError('stay probabilities must be in [0,1)')
    if hazard_kappa<0:raise ValueError('hazard_kappa must be nonnegative')
    if regret_reference=='soft_selector' and adversary_loss!='nll':raise ValueError('soft_selector requires nll')
    if regret_reference=='selector' and adversary_loss!='logical_error':raise ValueError('selector requires logical_error')
    if cost_units!='raw' and hc_cost_normalization!='hazard':raise ValueError('do not combine two cost-normalization policies')
    bank=get_case(case)
    start=time.perf_counter();torch.set_num_threads(1);spec=bank['specs'];src=np.asarray(bank['source']);tgt=np.asarray(bank['target']);K=len(src)
    if deployment_source_mass:
        if not 0<deployment_source_mass<1:raise ValueError('mass must be in (0,1)')
        src*=1-deployment_source_mass;src[-1]=deployment_source_mass
    def progress(s):print(f'[{profile} d{distance} {case}] {s}',flush=True)
    code,xy,edges,env,F,H,Z=setup(distance,spec,profile,strength,noise_geometry);ex=Experts(code,env,edges,F,H,Z,src,expert_calibration=expert_calibration,history=history,train_stay=train_stay,test_stay=test_stay)
    uniform=np.ones(K)/K
    # Independent pure-environment calibration for confidence, including known
    # zero-source-mass candidate; equally available to EVERY method.
    if calibration_per_env<10 or calibration_per_env%10:raise ValueError('calibration_per_env must be a positive multiple of 10')
    calibration_sequences=calibration_per_env//10
    parts=[draw(code,F,H,Z,np.eye(K)[k],calibration_sequences,10,314159+101*k,stay=train_stay) for k in range(K)]
    cal={key:np.concatenate([p[key] for p in parts]) for key in ['S','ERR','E']};cal.update(nseq=calibration_sequences*K,rounds=10,pi=uniform,stay=train_stay)
    ex.enrich(cal,calibrate=True);progress('independent physical expert calibration complete')
    def data(pi,nseq,sd,stay=None):
        d=ex.enrich(draw(code,F,H,Z,pi,nseq,14,sd,train_stay if stay is None else stay))
        return action_history(d) if feature_mode=='action_history' else d
    tr=data(uniform,ntrain,seed+100);va=data(uniform,nval,seed+1000);progress('shared full-trajectory training/validation data ready')
    C,geo=costs(ex,cal,src,witness_mode=witness_mode,kappa=hazard_kappa,regret_reference=regret_reference,reference_data=va,witness_sampling=witness_sampling)
    if hc_cost_normalization=='unit_median':
        positive=C['hc'][C['hc']>1e-12]
        if len(positive):C['hc']=C['hc']/np.median(positive)
    if cost_units!='raw':
        hc_scale=max(float(np.max(src@C['multiscale_ma'])),1e-12)
        for key in C:
            scale=hc_scale if key=='hc' and cost_units=='pre_hazard_bank' else max(float(np.max(src@C[key])),1e-12)
            C[key]=C[key]/scale
    geo['cost_units']=cost_units
    floor=-np.log(np.maximum(va['P'][np.arange(len(va['Y'])),va['E'],va['Y']],1e-10))
    requested=METHODS if methods is None else methods.split(',');models={};traces={}
    for m in requested:
        if m not in METHODS:raise ValueError(m)
        model,trace=fit(m,tr,va,src,C,floor,seed,epochs,rho,optimizer_mode,adversary_loss,adversary_search,train_stay,regret_reference,auxiliary_target,weight_normalization,checkpoint_mode);models[m]=model;traces[m]=trace;progress(m+' fitted')
    # Test is generated only after all model training is finished.
    te=data(tgt,ntest,seed+900000,test_stay);te['target_pi']=tgt;refs=ex.references(te);preds={};records={};gates={};last_controls={};last_predictions={}
    for m,model in models.items():
        P,G=prediction(model,te);pr=P.argmax(1);err=pr!=te['Y'];preds[m]=pr
        records[m]=dict(ler=float(err.mean()),repairs=int(np.sum((pr==te['Y'])&(te['Y']!=0))),breakages=int(np.sum((pr!=te['Y'])&(te['Y']==0))),nonzero=int(np.sum(pr!=0)),environment_accuracy=float(np.mean(G.argmax(1)==te['E'])))
        if checkpoint_mode=='late_average':
            LP,_=prediction(model.models[-1],te);lpr=LP.argmax(1);last_predictions[m]=lpr;last_controls[m]=dict(ler=float(np.mean(lpr!=te['Y'])),errors=int(np.sum(lpr!=te['Y'])),n=len(lpr))
    ci={};raw={m:(p!=te['Y']).astype(np.uint8).reshape(ntest,14) for m,p in preds.items()}
    if 'erm' in preds:
        for m in preds:
            g=(raw['erm'].astype(float)-raw[m]).mean(1);rng=np.random.default_rng(909001);boot=np.mean(g[rng.integers(0,ntest,(2000,ntest))],1)
            ci[m]=dict(gain=float(g.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),different_predictions=int(np.sum(preds[m]!=preds['erm'])))
    raw.update({'last__'+m:(p!=te['Y']).astype(np.uint8).reshape(ntest,14) for m,p in last_predictions.items()})
    audit=[]
    from canonical_noise import marginals
    for e in env:
        mx,mz=marginals(e,edges)
        audit.append(dict(mean_marginal_x=float(mx.mean()),mean_marginal_z=float(mz.mean()),local_mean=float(np.mean(e['px']+e['py']+e['pz'])),pair_endpoints_per_qubit=float(2*(e['pxx'].sum()+e['pzz'].sum())/code.n),max_local_rate=float(max(np.max(e[k]) for k in ['px','py','pz'])),max_pair_rate=float(max(np.max(e[k]) for k in ['pxx','pzz'])),burst_endpoints_per_qubit=0. if '_burst_matrix' not in e else float(np.asarray(e['_burst_matrix'].sum(0)).ravel()@e['_burst_prob']/code.n),burst_length=e.get('burst_length',2),burst_x_horizontal_share=e.get('burst_x_horizontal_share'),burst_z_horizontal_share=e.get('burst_z_horizontal_share')))
    result=dict(version='10.0',witness_sampling=witness_sampling,last_iterate_controls=last_controls,checkpoint_mode=checkpoint_mode,evaluation_role=evaluation_role,weight_normalization=weight_normalization,noise_geometry=noise_geometry,data_hashes={name:hashlib.sha256(d['S'].tobytes()+d['ERR'].tobytes()+d['E'].tobytes()).hexdigest() for name,d in [('train',tr),('validation',va),('test',te)]},regret_reference=regret_reference,auxiliary_target=auxiliary_target,feature_mode=feature_mode,cost_units=cost_units,implementation='corrected_shared_gru',suite=case,case_family=bank['family'],distance=distance,profile=profile,strength=1 if profile=='original' else strength,seed=seed,test_seed=seed+900000,architecture='non-GNN GRU regime gate + exact d3 / calibrated BP-OSD conditional experts',source_pi=src.tolist(),test_pi=tgt.tolist(),budgets=dict(ntrain=ntrain,nval=nval,epochs=epochs,ntest=ntest,test_rounds=14,confidence_calibration_per_env=calibration_per_env,history=history),methods=records,paired=ci,references={k:None if v is None else dict(ler=float(v.mean()),errors=int(v.sum()),n=len(v)) for k,v in refs.items()},reference_semantics=dict(corr_min='exact CSS component minimum-weight fault mechanism decoder' if ex.exact else 'approximate source-mechanism BP+OSD',source_hmm='exact finite-history HMM with exact conditional cosets' if ex.exact else 'factorized single-check pseudolikelihood HMM plus calibrated BP+OSD',bayes=f'exact target-prior history-{history} Bayes' if ex.exact else 'unavailable'),geometry=geo,costs={k:v.tolist() for k,v in C.items()},training_traces=traces,physics_audit=audit,resolved_noise_geometry=env[0].get('geometry_variant',noise_geometry),expert_confusion=ex.conf.tolist(),expert_calibration=expert_calibration,action_confusion=None if ex.action_conf is None else ex.action_conf.tolist(),optimizer_mode=optimizer_mode,adversary_loss=adversary_loss,adversary_search=adversary_search,hc_cost_normalization=hc_cost_normalization,rho=rho,hazard_kappa=hazard_kappa,train_stay=train_stay,test_stay=test_stay,modeled_support=bank["modeled_indices"],diagnostic_control_excluded=bank["control"] is not None,mass_on_diagnostic_control=0.0,elapsed_seconds=time.perf_counter()-start,
        limitations=['Known candidate bank includes deployment regime for every method','Single initialization/calibration seed; intervals condition on trained models','Exact d3 and approximate higher-distance physical experts differ',f'History is a {history}-step masked window; no GNN','Matched strength .6 is a new explicitly labeled benchmark, not a correction to measured original LER','Finite-bank, local Frank-Wolfe adversary; not globally certified nonconvex minimax'])
    # A persistent suite's sixth environment is never in training, calibration,
    # reference experts, geometry normalization, or the transport support.
    # Optional evaluation is post-training and is reported separately.
    if evaluate_control and bank['control'] is not None:
        _,_,_,_,_,_,Zc=setup(distance,[bank['control']],profile,strength,noise_geometry)
        control=draw(code,F,H,Zc,np.ones(1),100,14,seed+950000,test_stay)
        control['E'][:]=5
        ex.enrich(control)
        if feature_mode=='action_history':action_history(control)
        control_errors={}
        for method,model in models.items():
            pr,_=prediction(model,control);control_errors[method]=float(np.mean(pr.argmax(1)!=control['Y']))
        result['diagnostic_control']={'nominal_mass':0.,'test_mass_in_main_table':0.,'included_in_transport':False,'n':len(control['Y']),'mwpm_ler':float(np.mean(control['Y']!=0)),'methods':control_errors}
    return result,raw

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--calibration-per-env',type=int,default=600);ap.add_argument('--adversary-search',choices=['local','rays'],default='local');ap.add_argument('--case',required=True,choices=names());ap.add_argument('--distance',type=int,required=True,choices=[3,5,7,15,19]);ap.add_argument('--profile',choices=['matched','original'],default='matched');ap.add_argument('--strength',type=float,default=.6);ap.add_argument('--seed',type=int,default=107);ap.add_argument('--ntest',type=int,default=500);ap.add_argument('--ntrain',type=int,default=300);ap.add_argument('--nval',type=int,default=100);ap.add_argument('--epochs',type=int,default=24);ap.add_argument('--rho',type=float,default=.5);ap.add_argument('--optimizer-mode',choices=['iterative','one_shot'],default='iterative');ap.add_argument('--methods');ap.add_argument('--deployment-source-mass',type=float,default=0);ap.add_argument('--hc-cost-normalization',choices=['hazard','unit_median'],default='hazard');ap.add_argument('--witness-mode',choices=['disagreement','coset'],default='coset');ap.add_argument('--adversary-loss',choices=['nll','logical_error'],default='logical_error');ap.add_argument('--expert-calibration',choices=['unconditional','action'],default='action');ap.add_argument('--history',type=int,choices=[8,14],default=8);ap.add_argument('--train-stay',type=float,default=.95);ap.add_argument('--test-stay',type=float,default=.98);ap.add_argument('--hazard-kappa',type=float,default=3.);ap.add_argument('--evaluate-control',action='store_true');ap.add_argument('--out',required=True)
    ap.add_argument('--regret-reference',choices=['environment','selector','soft_selector'],default='environment');ap.add_argument('--auxiliary-target',choices=['environment','correct_expert'],default='environment');ap.add_argument('--feature-mode',choices=['legacy','action_history'],default='legacy');ap.add_argument('--cost-units',choices=['raw','bank','pre_hazard_bank'],default='raw')
    ap.add_argument('--noise-geometry',choices=['standard','ribbons','segments','long_segments','long_aligned','distance_scaled'],default='standard')
    ap.add_argument('--weight-normalization',choices=['none','self_normalized'],default='none')
    ap.add_argument('--evaluation-role',choices=['development','test'],default='test')
    ap.add_argument('--checkpoint-mode',choices=['last','late_average'],default='last')
    ap.add_argument('--witness-sampling',choices=['nominal','balanced'],default='nominal')
    a=vars(ap.parse_args());out=Path(a.pop('out'));r,raw=run(**a);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2));np.savez_compressed(out.with_suffix('.npz'),**raw);print('SAVED',out,flush=True)
