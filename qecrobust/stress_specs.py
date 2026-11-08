from __future__ import annotations

# Each environment is (family, local_pattern, pair_pattern, severity).
# Indices 0..3 are observed source environments; index 4 is the deployment-dominant
# candidate with zero nominal source mass; index 5 is a zero-mass benign control.
STRESS_SUITES = {
    'lcorner_family_topology': [
        ('mixed_pauli','l_tl','vertical_col0',1.0),
        ('mixed_pauli','hotspot_tl_2x2','vertical_col0',1.0),
        ('pauli_correlated','hotspot_top_row','vertical_col0',1.0),
        ('mixed_pauli','uniform','vertical_col0',1.0),
        ('crosstalk_pauli','l_bl','horizontal_row2',1.0),
        ('mixed_pauli','l_tr','vertical_col1',1.0),
    ],
    'checker_family_topology': [
        ('mixed_pauli','checker_even','perimeter',1.0),
        ('mixed_pauli','uniform','perimeter',1.0),
        ('pauli_correlated','hotspot_mid_col','perimeter',1.0),
        ('mixed_pauli','cross','perimeter',1.0),
        ('crosstalk_pauli','checker_odd','horizontal_all',1.0),
        ('mixed_pauli','uniform','vertical_col2',1.0),
    ],
    'gradient_family_topology': [
        ('pauli_correlated','gradient_lr','mixed_a',1.0),
        ('mixed_pauli','hotspot_right_col','mixed_a',1.0),
        ('pauli_correlated','uniform','mixed_a',1.0),
        ('mixed_pauli','l_br','mixed_a',1.0),
        ('crosstalk_pauli','gradient_rl','vertical_col0',1.0),
        ('mixed_pauli','l_bl','vertical_col1',1.0),
    ],
    'band_family_topology': [
        ('pauli_correlated','band_h_top','horizontal_all',1.0),
        ('mixed_pauli','gradient_bt','horizontal_all',1.0),
        ('pauli_correlated','hotspot_tr_2x2','horizontal_all',1.0),
        ('mixed_pauli','uniform','horizontal_all',1.0),
        ('crosstalk_pauli','band_v_right','vertical_all',1.0),
        ('mixed_pauli','gradient_tb','center_cross',1.0),
    ],
    'hotspot_family_topology': [
        ('mixed_pauli','hotspot_left_col','vertical_col0',1.0),
        ('pauli_correlated','gradient_rl','vertical_col0',1.0),
        ('mixed_pauli','uniform','vertical_col0',1.0),
        ('pauli_correlated','l_tl','vertical_col0',1.0),
        ('crosstalk_pauli','hotspot_right_col','horizontal_all',1.0),
        ('mixed_pauli','hotspot_left_col','vertical_all',1.0),
    ],
    'row_to_column_crosstalk': [
        ('crosstalk_pauli','uniform','vertical_col0',1.0),
        ('crosstalk_pauli','uniform','vertical_col1',1.0),
        ('crosstalk_pauli','uniform','vertical_col2',1.0),
        ('crosstalk_pauli','hotspot_mid_col','vertical_all',1.0),
        ('crosstalk_pauli','uniform','horizontal_all',1.0),
        ('crosstalk_pauli','uniform','center_cross',1.0),
    ],
    'mixed_to_correlated_checker': [
        ('mixed_pauli','checker_even','vertical_all',1.0),
        ('mixed_pauli','checker_odd','vertical_all',1.0),
        ('mixed_pauli','uniform','vertical_all',1.0),
        ('mixed_pauli','cross','vertical_all',1.0),
        ('pauli_correlated','checker_even','horizontal_all',1.0),
        ('mixed_pauli','checker_even','perimeter',1.0),
    ],
}

SOURCE_PI=(.03,.3233333333333333,.3233333333333333,.3233333333333333,0.,0.)
TEST_PI=(.0175,.0175,.0175,.0175,.93,0.)

# Environments 0..4 define the modeled ambiguity support. Environment 5 is a
# zero-mass diagnostic/benign control and must NEVER receive adversarial mass or
# participate in transport-geometry normalization/metric closure.
ROBUST_SUPPORT=(0,1,2,3,4)
CONTROL_INDEX=5
