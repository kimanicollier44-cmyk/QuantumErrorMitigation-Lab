"""Tests for QuantumErrorMitigation-Lab: circuit, ZNE, readout, PEC, gates."""
import math
from fractions import Fraction

import main as m


# ---- matrices, gates, density matrices
def test_gates_unitary():
    for kind, theta in m.CIRCUIT_GATES:
        u = m.gate_unitary(kind, theta)
        assert m.mat_close(m.mat_mul(u, m.mat_dagger(u)), m.IDENTITY)


def test_dm_basics():
    assert abs(m.mat_trace(m.RHO_0) - 1.0) < 1e-12
    assert abs(m.purity(m.RHO_0) - 1.0) < 1e-12
    assert abs(m.expect_z(m.RHO_0) - 1.0) < 1e-12
    assert abs(m.expect_z(m.dm_from_ket(m.KET_1)) + 1.0) < 1e-12


def test_single_gate_expectation():
    rho = m.run_circuit(0.0, gates=(("ry", math.pi / 3),))
    assert abs(m.expect_z(rho) - 0.5) < 1e-12


def test_ideal_in_range():
    assert 0.0 < m.ideal_expectation() < 1.0


# ---- noise channel
def test_channel_trace_and_purity():
    noisy = m.run_circuit(m.P_GATE)
    assert abs(m.mat_trace(noisy) - 1.0) < 1e-12
    assert m.purity(noisy) < m.purity(m.run_circuit(0.0)) - 0.01


def test_contraction_law():
    ideal = m.ideal_expectation()
    g = len(m.CIRCUIT_GATES)
    assert abs(m.expect_z(m.run_circuit(m.P_GATE))
               - (1 - m.P_GATE) ** g * ideal) < 1e-12
    for lam in m.ZNE_LAMBDAS:
        assert abs(m.expectation_at_lambda(lam)
                   - (1 - m.P_GATE) ** (lam * g) * ideal) < 1e-12


def test_scaled_noise_p():
    assert abs(m.scaled_noise_p(1) - m.P_GATE) < 1e-15
    assert abs(m.scaled_noise_p(2) - (1 - 0.95 ** 2)) < 1e-15


# ---- zero-noise extrapolation
def test_richardson_weights():
    w = m.richardson_weights()
    assert all(abs(a - b) < 1e-12 for a, b in zip(w, (3.0, -3.0, 1.0)))
    assert abs(sum(w) - 1.0) < 1e-12


def test_zne_mitigated_closer_not_exact():
    zne = m.zne_report()
    assert zne["raw_error"] > 0.05
    assert 0.0 < zne["mitigated_error"] < zne["raw_error"]
    assert zne["mitigated_error"] > 1e-6  # honest leftover bias
    assert zne["improvement_factor"] > 10.0


def test_zne_variance_factor():
    assert abs(m.zne_report()["variance_factor"] - 49.0) < 1e-9


# ---- readout mitigation
def test_confusion_columns():
    for j in range(2):
        assert abs(m.CONFUSION[0][j] + m.CONFUSION[1][j] - 1.0) < 1e-12


def test_readout_round_trip():
    rep = m.readout_report()
    assert rep["recovery_error"] < 1e-9
    assert abs(sum(rep["corrected"]) - 1.0) < 1e-12


def test_naive_inversion_negative():
    rep = m.readout_report()
    assert -0.01 < rep["naive_corrected"][1] < 0.0


def test_clip_policy():
    rep = m.readout_report()
    assert rep["clipped_corrected"] == (1.0, 0.0)
    assert abs(sum(rep["clipped_corrected"]) - 1.0) < 1e-12


# ---- probabilistic error cancellation
def test_pec_coefficients_exact():
    a, b = m.pec_coefficients()
    assert a == Fraction(79, 76)
    assert b == Fraction(-1, 76)


def test_pec_gamma():
    assert m.pec_gamma() == (2 + m.PEC_P) / (2 * (1 - m.PEC_P))
    assert m.pec_gamma() > 1
    assert m.pec_gamma(Fraction(1, 100)) < m.pec_gamma(Fraction(1, 20)) \
        < m.pec_gamma(Fraction(1, 10))


def test_pec_inverse_recovers():
    assert m.pec_report()["recovery_error"] < 1e-12


def test_pec_overhead_compounds():
    pec = m.pec_report()
    assert pec["gamma_squared_total"] > pec["gamma_squared_per_gate"] > 1.0


# ---- cost table
def test_cost_table():
    costs = m.cost_report()
    assert costs["zne_bias_before"] > costs["zne_bias_after"] > 0.0
    assert costs["zne_variance_factor"] > 1.0
    assert costs["readout_amplification"] > 1.0


# ---- classifier
def test_classifier_scenarios():
    rep = m.classifier_report()
    assert rep["demo_circuit_p0.05"] == "mitigate-with-ZNE"
    assert rep["gentle_noise_tiny_bias"] == "readout-only"
    assert rep["heavy_noise"] == "too-noisy-to-trust"
    assert rep["cheap_circuit_huge_overhead"] == "too-noisy-to-trust"


def test_classifier_matches_policy():
    for features in [(0.1025, 0.05, 1.84), (0.008, 0.01, 1.1),
                     (0.08, 0.05, 1.8), (0.18, 0.20, 5.0)]:
        assert m.classify_mitigation(*features) == m.policy_label(*features)


# ---- HOLD gates + council seats
def test_hold_gates_14():
    assert len(m.HOLD_GATES) == 14
    assert all(v.startswith("HOLD") for v in m.HOLD_GATES.values())


def test_gate_verdicts():
    assert m.gate("hardware_result_claim") == "HOLD"
    assert m.gate("mitigation_beats_error_correction_claim") == "HOLD"
    assert m.gate("suppress_raw_data") == "HOLD"
    assert m.gate("run_simulation") == "PROCEED"


def test_council_seats():
    brains = m.brains_report()
    assert brains["decider"]["seat"] == "Gemini Argon 4 Extended Thinking"
    assert brains["decider"]["can_override_hold"] is False
    assert brains["decider"]["live_calls"] == 0
    assert brains["advisor"]["can_override_hold"] is False
    assert brains["flow"]["rendered"] is False
    assert brains["flow"]["live"] == 0


def test_hold_demo_exit_code():
    assert m.main(["hold-demo"]) == 2
    assert m.main(["hold-demo", "run_simulation"]) == 0


def test_checks_all_pass():
    checks = m.run_checks()
    assert all(ok for _, ok in checks)
    assert len(checks) == 48
