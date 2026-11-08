"""Dependency-free study definitions. These presets are configurations, not results."""
from .case_bank import names
SEEDS5=[47,61,83,97,109]
SEEDS3=[47,61,83]
DISTANCES=[3,5,7,15,19]
BASE=dict(ntrain=300,nval=100,ntest=500,epochs=24,calibration_per_env=600,
          expert_calibration='action',adversary_loss='logical_error',witness_mode='coset',
          adversary_search='local',optimizer_mode='iterative',hc_cost_normalization='hazard',
          history=8,train_stay=.95,test_stay=.98,rho=.5,hazard_kappa=3.,strength=.6,
          deployment_source_mass=0.,evaluate_control=False,regret_reference='environment',auxiliary_target='environment',feature_mode='legacy',cost_units='raw',noise_geometry='standard',weight_normalization='none',evaluation_role='test',checkpoint_mode='last',witness_sampling='nominal')
PRESETS={
 'v9-final':dict(BASE,seed=107),
 'v9-intermediate':dict(BASE,seed=83,witness_mode='disagreement'),
 'v9-development':dict(BASE,seed=59,epochs=14,expert_calibration='unconditional',adversary_loss='nll',witness_mode='disagreement'),
 'v91-paper':dict(BASE,seed=151,ntrain=1200,nval=400,ntest=1000,calibration_per_env=7000,adversary_search='rays'),
}
PRESETS['v96-selector']=dict(BASE,seed=211,regret_reference='selector',auxiliary_target='correct_expert',cost_units='pre_hazard_bank')
PRESETS['v96-action-history']=dict(PRESETS['v96-selector'],feature_mode='action_history')
PRESETS['v96-soft-envelope']=dict(PRESETS['v96-action-history'],regret_reference='soft_selector',adversary_loss='nll')
PRESETS['v97-geometry-control']=dict(BASE,seed=619,noise_geometry='segments')
PRESETS['v97-geometry-candidate']=dict(PRESETS['v96-soft-envelope'],seed=619,noise_geometry='segments')
PRESETS['v97-segments']=dict(PRESETS['v97-geometry-control'])
PRESETS['v97-unit-control']=dict(PRESETS['v97-segments'],cost_units='pre_hazard_bank')
PRESETS['v97-stabilized']=dict(PRESETS['v97-unit-control'],weight_normalization='self_normalized')
PRESETS['v97-enhanced']=dict(PRESETS['v97-geometry-candidate'])
PRESETS['v98-screen']=dict(PRESETS['v97-unit-control'],seed=1493,ntrain=100,nval=80,ntest=120,epochs=12,calibration_per_env=150,evaluation_role='development')
PRESETS['v98-long']=dict(PRESETS['v97-unit-control'],noise_geometry='long_segments',seed=1879,adversary_search='rays',ntrain=160,nval=100,ntest=300,epochs=20,calibration_per_env=300)
PRESETS['v98-aligned']=dict(PRESETS['v97-unit-control'],noise_geometry='long_aligned',seed=1879,adversary_search='rays',ntrain=160,nval=100,ntest=300,epochs=20,calibration_per_env=300)
PRESETS['v98-long-average']=dict(PRESETS['v98-long'],checkpoint_mode='late_average')
PRESETS['v98-aligned-average']=dict(PRESETS['v98-aligned'],checkpoint_mode='late_average')
PRESETS['v98-long-balanced']=dict(PRESETS['v98-long'],witness_sampling='balanced')
PRESETS['v98-aligned-balanced']=dict(PRESETS['v98-aligned'],witness_sampling='balanced')
# Explicit d5 geometry control: preserves v98-long-average physics budget.
PRESETS['v99-d5-short']=dict(PRESETS['v98-long-average'],noise_geometry='segments')
PRESETS['v10-main']=dict(PRESETS['v98-long-average'],noise_geometry='distance_scaled')
PRESETS['v10-long-control']=dict(PRESETS['v98-long-average'])
PRESETS['v10-d5-short']=dict(PRESETS['v99-d5-short'])
# family, default distances, seed policy; case count multiplies these dimensions.
SUITES={
 'hv_pilot':('hv',[3],'one'),
 'hv_5seed':('hv',[3],'five'),
 'hv_d5_pilot':('hv',[5],'one'),
 'hv_d7_pilot':('hv',[7],'one'),
 'hv_d5d7_3seed':('hv',[5,7],'three'),
 'reproduce-quoted-matrix':('hv',DISTANCES,'one'),
 'hv_distance_pilot':('hv',DISTANCES,'one'),
 'hv_distance_5seed':('hv',DISTANCES,'five'),
 'persistent_pilot':('persistent',[3],'one'),
 'persistent_3seed':('persistent',[3],'three'),
 'persistent_5seed':('persistent',[3],'five'),
 'persistent_distance_5seed':('persistent',DISTANCES,'five'),
 'combined_pilot':('all',[3],'one'),
 'combined_5seed':('all',[3],'five'),
 'corrected_paper_core':('all',DISTANCES,'five'),
 'hv_band_pilot':('hv_band',[3],'one'),
 'hv_band_5seed':('hv_band',[3],'five'),
 'hv_band_distance_5seed':('hv_band',DISTANCES,'five'),
 'band_distance_5seed':('band',DISTANCES,'five'),
 'custom':('hv',[3],'one'),
 'corrected_preflight':('all',[3,5],'one'),
}
ALIASES={
 'hv_generalization_pilot':'hv_pilot','hv_generalization_5seed':'hv_5seed',
 'hv_distance_d5_pilot':'hv_d5_pilot','hv_distance_d7_pilot':'hv_d7_pilot',
 'hv_distance_d5d7_3seed':'hv_d5d7_3seed','complete_distance_pilot':'hv_distance_pilot',
 'distance_5seed':'hv_distance_5seed','persistent_correlated_stress_5seed':'persistent_5seed',
 'paper_core':'corrected_paper_core','paper_core_preflight':'corrected_preflight',
}

def suite_definition(name):
 canonical=ALIASES.get(name,name)
 if canonical not in SUITES:raise ValueError(f'unknown corrected suite {name!r}')
 return canonical,SUITES[canonical]

def preset(name):
 if name not in PRESETS:raise ValueError('unknown preset '+name)
 return dict(PRESETS[name])

def default_cases(family):return names(family)

def default_seeds(policy,configuration):
 return list(SEEDS5 if policy=='five' else SEEDS3 if policy=='three' else [configuration['seed']])
