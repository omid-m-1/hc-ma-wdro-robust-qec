"""Five modeled candidates per case; optional sixth benign diagnostic is excluded."""
from .headline_specs import HEADLINE_CASES,HEADLINE_SOURCE_PI,HEADLINE_TEST_PI
from .stress_specs import STRESS_SUITES,SOURCE_PI,TEST_PI,ROBUST_SUPPORT,CONTROL_INDEX
HV_ORDER=('hv_checker_cross_perimeter','hv_checker_even_center_cross','hv_hotspot_bottom_perimeter_control','hv_hotspot_cross_perimeter')
STRESS_ORDER=tuple(STRESS_SUITES)

def names(family='all'):
    if family=='band':return ['band_family_topology']
    if family=='hv_band':return list(HV_ORDER)+['band_family_topology']
    if family=='hv':return list(HV_ORDER)
    if family=='persistent':return list(STRESS_ORDER)
    if family=='all':return list(HV_ORDER)+list(STRESS_ORDER)
    raise ValueError('family must be hv, persistent or all')

def get_case(name):
    if name in HV_ORDER:
        return dict(name=name,family='hv',specs=list(HEADLINE_CASES[name]),source=list(HEADLINE_SOURCE_PI),target=list(HEADLINE_TEST_PI),control=None,modeled_indices=list(range(5)))
    if name in STRESS_SUITES:
        assert tuple(ROBUST_SUPPORT)==(0,1,2,3,4) and CONTROL_INDEX==5
        assert SOURCE_PI[CONTROL_INDEX]==0 and TEST_PI[CONTROL_INDEX]==0
        return dict(name=name,family='persistent',specs=[STRESS_SUITES[name][i] for i in ROBUST_SUPPORT],source=[SOURCE_PI[i] for i in ROBUST_SUPPORT],target=[TEST_PI[i] for i in ROBUST_SUPPORT],control=STRESS_SUITES[name][CONTROL_INDEX],modeled_indices=list(ROBUST_SUPPORT))
    raise ValueError(f'unknown case {name!r}; choose from '+','.join(names()))
