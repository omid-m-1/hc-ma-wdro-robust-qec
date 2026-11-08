from __future__ import annotations

# Exact development pilot cases frozen on 2026-11-06.
# Each case contains four positive-mass source environments followed by one
# zero-source-mass deployment-support environment that receives 93% test mass.
# Environment tuple: (family, local_pattern, pair_pattern, severity).
HEADLINE_CASES = {
    'hotspot_bottom_hall': [
        ('mixed_pauli','hotspot_left_col','vertical_col0',1.0),
        ('mixed_pauli','hotspot_mid_col','vertical_col0',1.0),
        ('pauli_correlated','hotspot_left_col','vertical_col1',1.0),
        ('mixed_pauli','uniform','vertical_col0',1.0),
        ('crosstalk_pauli','hotspot_bottom_row','horizontal_all',1.0),
    ],
    'hotspot_row2': [
        ('mixed_pauli','hotspot_left_col','vertical_col0',1.0),
        ('mixed_pauli','hotspot_mid_col','vertical_col0',1.0),
        ('pauli_correlated','hotspot_left_col','vertical_col1',1.0),
        ('mixed_pauli','uniform','vertical_col0',1.0),
        ('crosstalk_pauli','hotspot_right_col','horizontal_row2',1.0),
    ],
    'lcorner_row2': [
        ('mixed_pauli','l_tl','vertical_col0',1.0),
        ('mixed_pauli','l_tr','vertical_col0',1.0),
        ('pauli_correlated','l_tl','vertical_col1',1.0),
        ('mixed_pauli','uniform','vertical_col0',1.0),
        ('crosstalk_pauli','l_bl','horizontal_row2',1.0),
    ],
    'band_vright_perimeter': [
        ('pauli_correlated','band_h_top','horizontal_all',1.0),
        ('mixed_pauli','band_h_bottom','horizontal_all',1.0),
        ('pauli_correlated','band_v_left','vertical_all',1.0),
        ('mixed_pauli','uniform','horizontal_all',1.0),
        ('crosstalk_pauli','band_v_right','perimeter',1.0),
    ],

    'hv_checker_cross_perimeter': [
        ('mixed_pauli','checker_even','horizontal_all',1.0),
        ('mixed_pauli','checker_odd','vertical_all',1.0),
        ('pauli_correlated','cross','horizontal_all',1.0),
        ('mixed_pauli','uniform','vertical_all',1.0),
        ('crosstalk_pauli','cross','perimeter',1.0),
    ],
    'hv_hotspot_cross_perimeter': [
        ('mixed_pauli','hotspot_left_col','vertical_all',1.0),
        ('mixed_pauli','hotspot_top_row','horizontal_all',1.0),
        ('pauli_correlated','hotspot_mid_col','vertical_all',1.0),
        ('mixed_pauli','uniform','horizontal_all',1.0),
        ('crosstalk_pauli','cross','perimeter',1.0),
    ],
    'hv_checker_even_center_cross': [
        ('mixed_pauli','checker_even','horizontal_all',1.0),
        ('mixed_pauli','checker_odd','vertical_all',1.0),
        ('pauli_correlated','cross','horizontal_all',1.0),
        ('mixed_pauli','uniform','vertical_all',1.0),
        ('crosstalk_pauli','checker_even','center_cross',1.0),
    ],
    'hv_hotspot_bottom_perimeter_control': [
        ('mixed_pauli','hotspot_left_col','vertical_all',1.0),
        ('mixed_pauli','hotspot_top_row','horizontal_all',1.0),
        ('pauli_correlated','hotspot_mid_col','vertical_all',1.0),
        ('mixed_pauli','uniform','horizontal_all',1.0),
        ('crosstalk_pauli','hotspot_bottom_row','perimeter',1.0),
    ],
}

HEADLINE_LABELS = {
    'hotspot_bottom_hall': 'mixed/correlated hotspot-left/vertical -> crosstalk hotspot-bottom/horizontal-all',
    'hotspot_row2': 'mixed/correlated hotspot-left/vertical -> crosstalk hotspot-right/horizontal-row2',
    'lcorner_row2': 'mixed/correlated L-corner/vertical -> crosstalk L-corner-BL/horizontal-row2',
    'band_vright_perimeter': 'correlated/mixed band -> crosstalk band-v-right/perimeter',

    'hv_checker_cross_perimeter': 'checker source H+V -> crosstalk cross + perimeter H+V',
    'hv_hotspot_cross_perimeter': 'hotspot source H+V -> crosstalk cross + perimeter H+V',
    'hv_checker_even_center_cross': 'checker source H+V -> crosstalk checker-even + center-cross H+V',
    'hv_hotspot_bottom_perimeter_control': 'hotspot source H+V -> crosstalk hotspot-bottom + perimeter H+V',
}

HEADLINE_SOURCE_PI=(.03,.3233333333333333,.3233333333333333,.3233333333333333,0.0)
HEADLINE_TEST_PI=(.0175,.0175,.0175,.0175,.93)
