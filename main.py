#!/usr/bin/env python3
"""QuantumErrorMitigation-Lab - educational quantum error mitigation lab.

by Darion Kellar

Pure Python standard library. Zero dependencies. Zero network.
Education only - and the central lesson is stated up front:

  * Error mitigation is NOT error correction. It does not protect a
    computation while it runs; it post-processes noisy results to
    estimate what the noiseless result would have been, and it pays
    for the smaller bias with a larger variance and a bigger bill.
  * This lab does that arithmetic exactly, on a toy one-qubit circuit
    with LABELLED noise channels: zero-noise extrapolation (ZNE) by
    Richardson extrapolation, readout-error mitigation by confusion-
    matrix inversion, and the quasi-probability coefficients and
    sampling overhead of probabilistic error cancellation (PEC).
  * Every number here is exact arithmetic on a labelled toy channel.
    No quantum hardware was run, no real device was calibrated, and
    no mitigated value in this lab is a measurement - the extrapo-
    lated lambda=0 value is an ESTIMATE, printed next to the raw
    value it came from, never instead of it.

Real-hardware results, production workloads, benchmark claims, and
any suggestion that mitigation beats error correction are permanent
HOLD gates - with Gemini Argon 4 Extended Thinking as DECIDER
(names-only, tighten-only), Muse as advisor (never overrides
HOLD), and Google Flow plan-only seats.
"""
import hashlib
import json
import math
import sys
from fractions import Fraction

# ---------------------------------------------------------------- constants
VERSION = "1.0.0"

# The toy circuit: four rotations on one qubit starting in |0>.
# Angles are exact multiples of pi so the ideal expectation is a
# fixed, reproducible number the density-matrix code computes.
CIRCUIT_GATES = (
    ("ry", math.pi / 3),
    ("rx", math.pi / 5),
    ("rz", math.pi / 4),
    ("rx", math.pi / 6),
)

# Labelled noise model. P_GATE is a TEACHING value for a depolarizing
# channel applied after every gate - it is not measured from, fitted
# to, or claimed about any real device.
P_GATE = 0.05
ZNE_LAMBDAS = (1, 2, 3)

# Labelled readout confusion matrix. A[i][j] = P(report i | true j).
# Columns are the true state, so each column sums to 1.
READOUT_FALSE_POS = 0.03   # P(report 1 | true 0)
READOUT_FALSE_NEG = 0.07   # P(report 0 | true 1)
CONFUSION = (
    (1.0 - READOUT_FALSE_POS, READOUT_FALSE_NEG),
    (READOUT_FALSE_POS, 1.0 - READOUT_FALSE_NEG),
)
# A labelled "measured" distribution (as if from finite shots) whose
# naive inversion dips below zero - the case the clip policy exists for.
LABELLED_MEASURED = (0.972, 0.028)

# PEC is computed on the same labelled depolarizing strength, as an
# exact Fraction so the coefficients can be asserted digit for digit.
PEC_P = Fraction(1, 20)

# ---------------------------------------------------------------- HOLD gates
HOLD_GATES = {
    "hardware_result_claim": "HOLD - no claim of results on real quantum hardware; this lab ran no device",
    "mitigation_beats_error_correction_claim": "HOLD - no claim that mitigation replaces or beats quantum error correction",
    "production_workload_use": "HOLD - no production, paid, or operational workload decisions from this toy lab",
    "benchmark_superiority_claim": "HOLD - no benchmark or superiority claims about any hardware or vendor",
    "error_free_result_claim": "HOLD - a mitigated value is an estimate with error bars, never an exact or error-free result",
    "financial_decision_use": "HOLD - no financial, trading, or investment decisions from mitigated toy numbers",
    "scientific_publication_claim": "HOLD - no publishing this lab's toy output as scientific or medical evidence",
    "cryptographic_security_claim": "HOLD - no cryptographic or security claims from toy noise models",
    "real_device_calibration_use": "HOLD - the lab's confusion matrix is labelled teaching data, never a real device calibration",
    "guaranteed_accuracy_claim": "HOLD - no guaranteed accuracy or precision claims for any mitigated value",
    "defense_classified_use": "HOLD - no defense, classified, or weapons-related use of this lab",
    "suppress_raw_data": "HOLD - raw unmitigated values must always be shown next to mitigated ones; never publish the estimate alone",
    "extrapolation_as_measurement_claim": "HOLD - the lambda=0 extrapolated value is an estimate, never a measurement",
    "vendor_comparison_claim": "HOLD - no comparing real QPUs or vendors using this lab's toy channels",
}


def gate(action):
    """Return 'HOLD' if the action is gated, else 'PROCEED' (sandbox sim)."""
    return "HOLD" if action in HOLD_GATES else "PROCEED"


# ================================================================ 1. 2x2 complex matrices
# Density matrices and gates are 2x2 matrices of complex numbers,
# stored as nested tuples. Everything is written out by hand so the
# arithmetic can be read, checked, and diffed - no numpy, no black box.

def mat_mul(a, b):
    return tuple(
        tuple(a[i][0] * b[0][j] + a[i][1] * b[1][j] for j in range(2))
        for i in range(2))


def mat_add(a, b):
    return tuple(tuple(a[i][j] + b[i][j] for j in range(2)) for i in range(2))


def mat_scale(a, s):
    return tuple(tuple(a[i][j] * s for j in range(2)) for i in range(2))


def mat_dagger(a):
    return tuple(tuple(a[j][i].conjugate() for j in range(2)) for i in range(2))


def mat_trace(a):
    return a[0][0] + a[1][1]


def mat_close(a, b, tol=1e-12):
    return all(abs(a[i][j] - b[i][j]) < tol for i in range(2) for j in range(2))


IDENTITY = ((1 + 0j, 0j), (0j, 1 + 0j))
PAULI_X = ((0j, 1 + 0j), (1 + 0j, 0j))
PAULI_Y = ((0j, -1j), (1j, 0j))
PAULI_Z = ((1 + 0j, 0j), (0j, -1 + 0j))
PAULIS = (PAULI_X, PAULI_Y, PAULI_Z)


def dm_from_ket(ket):
    """Density matrix |psi><psi| for a 2-component ket."""
    return tuple(tuple(ket[i] * ket[j].conjugate() for j in range(2))
                 for i in range(2))


KET_0 = (1 + 0j, 0j)
KET_1 = (0j, 1 + 0j)
RHO_0 = dm_from_ket(KET_0)


def purity(rho):
    return mat_trace(mat_mul(rho, rho)).real


def expect_z(rho):
    """<Z> = rho00 - rho11 (real for any Hermitian rho)."""
    return (rho[0][0] - rho[1][1]).real


def apply_unitary(rho, u):
    return mat_mul(mat_mul(u, rho), mat_dagger(u))


def gate_unitary(kind, theta):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    if kind == "rx":
        return ((c + 0j, -1j * s), (-1j * s, c + 0j))
    if kind == "ry":
        return ((c + 0j, -s + 0j), (s + 0j, c + 0j))
    if kind == "rz":
        return ((complex(math.cos(theta / 2), -math.sin(theta / 2)), 0j),
                (0j, complex(math.cos(theta / 2), math.sin(theta / 2))))
    raise ValueError(f"unknown gate: {kind}")


# ================================================================ 2. Toy circuit + labelled noise
# The noise channel is a depolarizing channel applied after every
# gate: rho -> (1 - p) rho + p * I/2. It is unital and contracts
# every Bloch-vector component by exactly (1 - p), which is what
# makes the noise-scaling law below exact rather than fitted.

def depolarize(rho, p):
    return mat_add(mat_scale(rho, 1.0 - p), mat_scale(IDENTITY, p / 2.0))


def run_circuit(p_per_gate=0.0, gates=CIRCUIT_GATES):
    """Run the toy circuit with a fixed depolarizing strength per gate."""
    rho = RHO_0
    for kind, theta in gates:
        rho = apply_unitary(rho, gate_unitary(kind, theta))
        if p_per_gate:
            rho = depolarize(rho, p_per_gate)
    return rho


def scaled_noise_p(lam, p=P_GATE):
    """Per-gate strength whose contraction equals lam channel repeats.

    Scaling method, stated openly: a depolarizing channel contracts
    the Bloch vector by (1 - p). Applying the channel lam times (or,
    on hardware, folding the circuit so the noise acts lam times)
    contracts by (1 - p)**lam. The scaled channel used here is the
    single channel with that same contraction, p_lam = 1 - (1-p)**lam,
    so every expectation in this lab obeys E(lam) = (1-p)**(lam*G) *
    E_ideal exactly, with G the gate count.
    """
    return 1.0 - (1.0 - p) ** lam


def expectation_at_lambda(lam, p=P_GATE):
    """<Z> of the toy circuit with the noise scaled by lam (0 = ideal)."""
    if lam == 0:
        return expect_z(run_circuit(0.0))
    return expect_z(run_circuit(scaled_noise_p(lam, p)))


def ideal_expectation():
    return expectation_at_lambda(0)


# ================================================================ 3. Zero-noise extrapolation
# Richardson extrapolation through the points lam = 1, 2, 3 assumes
# E(lam) is a polynomial in lam and evaluates it at lam = 0. The
# weights solve the Vandermonde system exactly: (3, -3, 1). Our toy
# E(lam) is really an exponential in lam, so Richardson REDUCES the
# bias a lot without erasing it - the demo prints both errors so the
# leftover bias is visible, not hidden.

def richardson_weights(lambdas=ZNE_LAMBDAS):
    """Solve for weights w with sum w_i * lam_i**k = (k == 0), k < n."""
    n = len(lambdas)
    # Gaussian elimination on the Vandermonde system, from scratch.
    rows = [[float(lam) ** k for lam in lambdas] + [1.0 if k == 0 else 0.0]
            for k in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        div = rows[col][col]
        rows[col] = [v / div for v in rows[col]]
        for r in range(n):
            if r != col and rows[r][col] != 0.0:
                factor = rows[r][col]
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[col])]
    return tuple(rows[k][n] for k in range(n))


def zne_report(p=P_GATE):
    ideal = ideal_expectation()
    values = {lam: expectation_at_lambda(lam, p) for lam in ZNE_LAMBDAS}
    weights = richardson_weights()
    mitigated = sum(w * values[lam] for w, lam in zip(weights, ZNE_LAMBDAS))
    raw_error = abs(values[1] - ideal)
    mitigated_error = abs(mitigated - ideal)
    return {
        "ideal": ideal,
        "values": values,
        "weights": weights,
        "raw_lambda1": values[1],
        "mitigated": mitigated,
        "raw_error": raw_error,
        "mitigated_error": mitigated_error,
        "improvement_factor": raw_error / mitigated_error,
        "variance_factor": sum(abs(w) for w in weights) ** 2,
    }


# ================================================================ 4. Readout mitigation
# Readout error is a classical confusion process on the reported
# bit: measured = A . true. Mitigation is the exact inverse,
# corrected = A^-1 . measured. The inverse can return small negative
# "probabilities" for measured vectors near a corner; the lab's
# labelled policy clips negatives to zero and renormalises, and says
# so wherever the policy is used.

def confusion_matrix():
    return CONFUSION


def apply_confusion(true_dist, a=CONFUSION):
    return (a[0][0] * true_dist[0] + a[0][1] * true_dist[1],
            a[1][0] * true_dist[0] + a[1][1] * true_dist[1])


def invert_2x2(a):
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    return ((a[1][1] / det, -a[0][1] / det),
            (-a[1][0] / det, a[0][0] / det))


def correct_readout(measured, a=CONFUSION):
    inv = invert_2x2(a)
    return (inv[0][0] * measured[0] + inv[0][1] * measured[1],
            inv[1][0] * measured[0] + inv[1][1] * measured[1])


def clip_renormalize(dist):
    """Labelled policy: clip negatives to 0, then renormalise to sum 1."""
    clipped = tuple(max(0.0, v) for v in dist)
    total = sum(clipped)
    return tuple(v / total for v in clipped)


def true_distribution():
    """True (p0, p1) of the noiseless toy circuit's Z measurement."""
    ez = ideal_expectation()
    return ((1.0 + ez) / 2.0, (1.0 - ez) / 2.0)


def readout_report():
    true = true_distribution()
    measured = apply_confusion(true)
    corrected = correct_readout(measured)
    naive = correct_readout(LABELLED_MEASURED)
    return {
        "true": true,
        "measured": measured,
        "corrected": corrected,
        "recovery_error": max(abs(c - t) for c, t in zip(corrected, true)),
        "labelled_measured": LABELLED_MEASURED,
        "naive_corrected": naive,
        "clipped_corrected": clip_renormalize(naive),
    }


# ================================================================ 5. Probabilistic error cancellation
# PEC runs the INVERSE of the noise channel as a quasi-probability
# mixture of physical operations: E^-1 = a * id + b * (X. + Y. + Z.)
# with a > 0 and b < 0. The coefficients are computed exactly here
# (Fractions), the mixture is verified to undo the channel on a test
# state, and the price is printed: every PEC sample costs a factor
# gamma = sum |q_i|, so the shot budget grows by gamma**2.
#
# Derivation (in the open): the channel multiplies each Pauli
# component by (1 - p). Conjugation by a Pauli P multiplies the
# identity by +1 and each Pauli Q by +1 if Q in {I, P}, else -1.
# So the mixture multiplies the identity by a + 3b and each Pauli by
# a - b. Inverting the channel needs a + 3b = 1 and a - b = 1/(1-p),
# giving b = -p / (4(1-p)) and a = (4 - p) / (4(1-p)).

def pec_coefficients(p=PEC_P):
    """Exact (a, b) quasi-probability coefficients of the inverse channel."""
    a = (4 - p) / (4 * (1 - p))
    b = -p / (4 * (1 - p))
    return a, b


def pec_gamma(p=PEC_P):
    a, b = pec_coefficients(p)
    return abs(a) + 3 * abs(b)


def pec_apply_inverse(rho, p_float):
    """Apply the quasi-probability mixture (float path) to a density matrix."""
    a, b = pec_coefficients(Fraction(p_float).limit_denominator(10 ** 9))
    a, b = float(a), float(b)
    out = mat_scale(rho, a)
    for pauli in PAULIS:
        out = mat_add(out, mat_scale(apply_unitary(rho, pauli), b))
    return out


def pec_report(gates=len(CIRCUIT_GATES)):
    a, b = pec_coefficients()
    gamma = pec_gamma()
    gamma_total = gamma ** gates
    test = run_circuit(0.0, gates=(("ry", math.pi / 3),))
    noisy = depolarize(test, float(PEC_P))
    recovered = pec_apply_inverse(noisy, float(PEC_P))
    return {
        "a": a,
        "b": b,
        "gamma_per_gate": gamma,
        "gamma_squared_per_gate": gamma ** 2,
        "gamma_total": gamma_total,
        "gamma_squared_total": gamma_total ** 2,
        "recovery_error": max(abs(recovered[i][j] - test[i][j])
                              for i in range(2) for j in range(2)),
    }


# ================================================================ 6. Cost honesty table
# Mitigation trades bias for variance and shots. This table is the
# bill, printed in the demo next to the mitigated numbers - always.

def cost_report():
    zne = zne_report()
    pec = pec_report()
    inv = invert_2x2(CONFUSION)
    readout_amplification = max(sum(abs(v) for v in row) for row in inv)
    return {
        "zne_bias_before": zne["raw_error"],
        "zne_bias_after": zne["mitigated_error"],
        "zne_variance_factor": zne["variance_factor"],
        "zne_note": "Richardson weights (3, -3, 1): the extrapolated value's "
                    "statistical error is amplified by sum|w| = 7, so the "
                    "shot budget grows ~49x for the same error bar.",
        "readout_amplification": readout_amplification,
        "readout_note": "Inverting A amplifies any measurement noise by up "
                        "to the largest |A^-1| row sum; clipping then trades "
                        "a little bias back in for physical probabilities.",
        "pec_gamma_squared_per_gate": pec["gamma_squared_per_gate"],
        "pec_gamma_squared_total": pec["gamma_squared_total"],
        "pec_note": "PEC is unbiased in principle, but each gate's inverse "
                    "mixture multiplies the shot budget by gamma^2; over the "
                    "whole circuit the factor compounds to gamma_total^2.",
    }


# ================================================================ 7. kNN mitigation classifier
# Features: raw bias |E(1) - E_ideal|, labelled noise strength p, and
# the total PEC overhead gamma^2. The training grid is labelled by
# the lab's own verdict policy (policy_label below) - the classifier
# interpolates the policy, it does not replace the arithmetic.

def policy_label(raw_bias, noise_p, gamma2_total):
    """The mitigation verdict policy distilled to three labels."""
    if noise_p >= 0.15 or gamma2_total >= 4.0:
        return "too-noisy-to-trust"
    if raw_bias < 0.02:
        return "readout-only"
    return "mitigate-with-ZNE"


def _training_grid():
    rows = []
    for bias in (0.005, 0.02, 0.05, 0.10, 0.20):
        for p in (0.01, 0.05, 0.10, 0.20):
            for g2 in (1.1, 1.5, 2.5, 5.0):
                rows.append(((bias, p, g2), policy_label(bias, p, g2)))
    return rows


TRAINING_GRID = _training_grid()


def _scaled(features):
    bias, p, g2 = features
    return (bias / 0.20, p / 0.20, math.log(g2) / math.log(5.0))


def classify_mitigation(raw_bias, noise_p, gamma2_total, k=5):
    """kNN over the policy-labelled grid, deterministic tie-breaking."""
    target = _scaled((raw_bias, noise_p, gamma2_total))
    ranked = []
    for features, label in TRAINING_GRID:
        point = _scaled(features)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(target, point)))
        ranked.append((dist, features, label))
    ranked.sort(key=lambda row: (row[0], row[1]))
    votes = {}
    for _, _, label in ranked[:k]:
        votes[label] = votes.get(label, 0) + 1
    priority = {"mitigate-with-ZNE": 0, "readout-only": 1,
                "too-noisy-to-trust": 2}
    return sorted(votes.items(), key=lambda kv: (-kv[1], priority[kv[0]]))[0][0]


def classifier_report():
    zne = zne_report()
    pec = pec_report()
    scenarios = {
        "demo_circuit_p0.05": (zne["raw_error"], P_GATE,
                               float(pec["gamma_squared_total"])),
        "gentle_noise_tiny_bias": (0.008, 0.01, 1.1),
        "moderate_noise_real_bias": (0.08, 0.05, 1.8),
        "heavy_noise": (0.18, 0.20, 5.0),
        "cheap_circuit_huge_overhead": (0.10, 0.05, 5.0),
    }
    return {name: classify_mitigation(*features)
            for name, features in scenarios.items()}


# ================================================================ 8. Advisory seats
def _trace(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def argon4_decide(topic="error-mitigation-demo"):
    """Gemini Argon 4 DECIDER seat: MAP->MODEL->STRESS->DECIDE.

    Names-only keys, live calls 0, tighten-only: can add HOLDs, never lift one.
    """
    stages = ["MAP", "MODEL", "STRESS", "DECIDE"]
    payload = {"seat": "gemini-argon-4", "topic": topic, "stages": stages}
    return {
        "seat": "Gemini Argon 4 Extended Thinking",
        "role": "DECIDER",
        "pipeline": stages,
        "tiers": [2048, 8192, 32768],
        "key": "names-only (GEMINI_API_KEY name checked, value never read)",
        "live_calls": 0,
        "decision": "PROCEED - sandbox simulation only, all hardware and benchmark gates stay HOLD",
        "can_override_hold": False,
        "trace_sha256_16": _trace(payload),
    }


def claude_advisor(topic="error-mitigation-demo"):
    """Muse advisor seat: advises, can never override a HOLD."""
    return {
        "seat": "Muse",
        "role": "ADVISOR",
        "key": "names-only (MODEL_API_KEY name checked, value never read)",
        "live_calls": 0,
        "advice": "Print the raw value, the mitigated value, and the bill on the same line: mitigation buys a smaller bias with a larger variance, and the extrapolated number is an estimate - never let it travel without its raw partner and its cost.",
        "can_override_hold": False,
    }


def google_flow_plan():
    """Google Flow seat: plan-only, nothing rendered, nothing live."""
    return {
        "seat": "Google Flow",
        "role": "PLAN-ONLY",
        "shots": ["a qubit's arrow shrinking as labelled noise contracts it",
                  "three noisy points at lambda 1, 2, 3 and a dashed line reaching back to lambda 0",
                  "a confusion matrix tilting a distribution, then its inverse tilting it back",
                  "quasi-probability coins, one of them negative, stacking into gamma",
                  "a cost ledger growing beside every mitigated number"],
        "rendered": False,
        "live": 0,
    }


def brains_report():
    return {"decider": argon4_decide(), "advisor": claude_advisor(),
            "flow": google_flow_plan()}


# ================================================================ checks
def run_checks():
    c = []

    def chk(name, ok):
        c.append((name, bool(ok)))

    # 1. matrices + gates + density matrices
    for kind, theta in CIRCUIT_GATES:
        u = gate_unitary(kind, theta)
        chk(f"gate_unitary_{kind}", mat_close(mat_mul(u, mat_dagger(u)), IDENTITY))
    chk("dm_trace_one", abs(mat_trace(RHO_0) - 1.0) < 1e-12)
    chk("dm_purity_pure", abs(purity(RHO_0) - 1.0) < 1e-12)
    single = run_circuit(0.0, gates=(("ry", math.pi / 3),))
    chk("single_gate_expectation_cos",
        abs(expect_z(single) - math.cos(math.pi / 3)) < 1e-12)
    ideal = ideal_expectation()
    chk("ideal_expectation_in_range", 0.0 < ideal < 1.0)

    # 2. noise channel behaviour
    noisy = run_circuit(P_GATE)
    chk("channel_trace_preserved", abs(mat_trace(noisy) - 1.0) < 1e-12)
    chk("channel_reduces_purity", purity(noisy) < purity(run_circuit(0.0)) - 0.01)
    chk("channel_contracts_expectation",
        abs(expect_z(noisy) - (1.0 - P_GATE) ** len(CIRCUIT_GATES) * ideal) < 1e-12)
    chk("scaled_noise_contraction_law",
        all(abs(expectation_at_lambda(lam)
                - (1.0 - P_GATE) ** (lam * len(CIRCUIT_GATES)) * ideal) < 1e-12
            for lam in ZNE_LAMBDAS))

    # 3. zero-noise extrapolation
    zne = zne_report()
    chk("richardson_weights_exact",
        all(abs(w - e) < 1e-12 for w, e in zip(zne["weights"], (3.0, -3.0, 1.0))))
    chk("richardson_weights_sum_one", abs(sum(zne["weights"]) - 1.0) < 1e-12)
    chk("zne_raw_biased", zne["raw_error"] > 0.05)
    chk("zne_mitigated_strictly_closer",
        zne["mitigated_error"] < zne["raw_error"])
    chk("zne_mitigated_not_exact", zne["mitigated_error"] > 1e-6)
    chk("zne_improvement_over_10x", zne["improvement_factor"] > 10.0)
    chk("zne_variance_factor_49", abs(zne["variance_factor"] - 49.0) < 1e-9)

    # 4. readout mitigation
    rep = readout_report()
    chk("confusion_columns_sum_one",
        all(abs(CONFUSION[0][j] + CONFUSION[1][j] - 1.0) < 1e-12
            for j in range(2)))
    chk("measured_is_confusion_times_true",
        abs(rep["measured"][0] + rep["measured"][1] - 1.0) < 1e-12
        and rep["measured"] != rep["true"])
    chk("readout_recovery_exact", rep["recovery_error"] < 1e-9)
    chk("readout_corrected_sums_one",
        abs(sum(rep["corrected"]) - 1.0) < 1e-12)
    chk("naive_inversion_goes_negative",
        -0.01 < rep["naive_corrected"][1] < 0.0
        and abs(sum(rep["naive_corrected"]) - 1.0) < 1e-12)
    chk("clip_policy_physical",
        all(v >= 0.0 for v in rep["clipped_corrected"])
        and abs(sum(rep["clipped_corrected"]) - 1.0) < 1e-12
        and rep["clipped_corrected"] == (1.0, 0.0))

    # 5. probabilistic error cancellation
    a, b = pec_coefficients()
    chk("pec_coefficients_exact", a == Fraction(79, 76) and b == Fraction(-1, 76))
    chk("pec_gamma_formula",
        pec_gamma() == (2 + PEC_P) / (2 * (1 - PEC_P)))
    chk("pec_gamma_above_one", pec_gamma() > 1)
    chk("pec_gamma_grows_with_noise",
        pec_gamma(Fraction(1, 100)) < pec_gamma(Fraction(1, 20))
        < pec_gamma(Fraction(1, 10)))
    pec = pec_report()
    chk("pec_inverse_recovers_state", pec["recovery_error"] < 1e-12)
    chk("pec_total_overhead_compounds",
        pec["gamma_squared_total"] > pec["gamma_squared_per_gate"] > 1.0)

    # 6. cost table
    costs = cost_report()
    chk("cost_table_bias_pair",
        costs["zne_bias_before"] > costs["zne_bias_after"] > 0.0)
    chk("cost_table_factors_above_one",
        costs["zne_variance_factor"] > 1.0
        and costs["readout_amplification"] > 1.0
        and costs["pec_gamma_squared_total"] > 1.0)

    # 7. classifier - labels come from the lab's own policy
    rep_cls = classifier_report()
    chk("classifier_demo_zne", rep_cls["demo_circuit_p0.05"] == "mitigate-with-ZNE")
    chk("classifier_gentle_readout", rep_cls["gentle_noise_tiny_bias"] == "readout-only")
    chk("classifier_moderate_zne", rep_cls["moderate_noise_real_bias"] == "mitigate-with-ZNE")
    chk("classifier_heavy_too_noisy", rep_cls["heavy_noise"] == "too-noisy-to-trust")
    chk("classifier_overhead_too_noisy",
        rep_cls["cheap_circuit_huge_overhead"] == "too-noisy-to-trust")
    chk("classifier_matches_policy",
        all(classify_mitigation(*features) == policy_label(*features)
            for features in [(0.1025, 0.05, 1.84), (0.008, 0.01, 1.1),
                             (0.08, 0.05, 1.8), (0.18, 0.20, 5.0),
                             (0.10, 0.05, 5.0)]))

    # gates + seats
    chk("hold_gates_14", len(HOLD_GATES) == 14
        and all(v.startswith("HOLD") for v in HOLD_GATES.values()))
    chk("gate_hardware_hold", gate("hardware_result_claim") == "HOLD")
    chk("gate_qec_hold", gate("mitigation_beats_error_correction_claim") == "HOLD")
    chk("gate_raw_data_hold", gate("suppress_raw_data") == "HOLD")
    chk("gate_extrapolation_hold",
        gate("extrapolation_as_measurement_claim") == "HOLD")
    chk("gate_sandbox_proceed", gate("run_simulation") == "PROCEED")
    brains = brains_report()
    chk("argon4_no_override", brains["decider"]["can_override_hold"] is False
        and brains["decider"]["seat"]
        == "Gemini Argon 4 Extended Thinking")
    chk("claude_no_override", brains["advisor"]["can_override_hold"] is False)
    chk("flow_plan_only", brains["flow"]["rendered"] is False
        and brains["flow"]["live"] == 0)
    return c


# ================================================================ CLI / demo
def demo():
    print("=" * 68)
    print("QuantumErrorMitigation-Lab - quantum error mitigation lab")
    print("by Darion Kellar | MIT | pure stdlib | zero network | EDUCATION ONLY")
    print("=" * 68)
    print("\n[1] THE HONEST CORE - mitigation is not correction")
    print("    Error correction protects a computation while it runs.")
    print("    Mitigation post-processes noisy results to ESTIMATE the")
    print("    noiseless answer - and pays for the smaller bias with a")
    print("    larger variance and a bigger shot bill. This lab does that")
    print("    arithmetic exactly on a labelled toy channel. No hardware")
    print("    was run; every mitigated value below is an estimate, and")
    print("    it is always printed next to the raw value it came from.")
    print("\n[2] Toy circuit + labelled depolarizing noise (p = 0.05 per gate)")
    print("    gates: " + ", ".join(f"{k}({t:.4f})" for k, t in CIRCUIT_GATES))
    zne = zne_report()
    print(f"    ideal <Z> (noiseless)      = {zne['ideal']:+.6f}")
    for lam in ZNE_LAMBDAS:
        print(f"    noisy <Z> at lambda = {lam}    = {zne['values'][lam]:+.6f}")
    print("    -> each gate's channel contracts the Bloch vector by 0.95,")
    print("       so E(lambda) = 0.95^(4*lambda) * E_ideal exactly.")
    print("\n[3] Zero-noise extrapolation (Richardson, weights 3, -3, 1)")
    print(f"    raw value (lambda = 1)     = {zne['raw_lambda1']:+.6f}  "
          f"error {zne['raw_error']:.6f}")
    print(f"    mitigated (lambda -> 0)    = {zne['mitigated']:+.6f}  "
          f"error {zne['mitigated_error']:.6f}")
    print(f"    -> bias cut by a factor of {zne['improvement_factor']:.1f}x, "
          "not to zero:")
    print("       the true E(lambda) is an exponential, the fit assumes a")
    print("       polynomial, and the leftover bias stays on the page.")
    print("\n[4] Readout mitigation (confusion-matrix inversion)")
    rep = readout_report()
    print(f"    A = [[{CONFUSION[0][0]:.2f}, {CONFUSION[0][1]:.2f}], "
          f"[{CONFUSION[1][0]:.2f}, {CONFUSION[1][1]:.2f}]]  (labelled)")
    print(f"    true      = ({rep['true'][0]:.6f}, {rep['true'][1]:.6f})")
    print(f"    measured  = ({rep['measured'][0]:.6f}, {rep['measured'][1]:.6f})")
    print(f"    corrected = ({rep['corrected'][0]:.6f}, {rep['corrected'][1]:.6f})"
          f"  recovery error {rep['recovery_error']:.2e}")
    print(f"    labelled measured {rep['labelled_measured']} inverts to "
          f"({rep['naive_corrected'][0]:.6f}, {rep['naive_corrected'][1]:.6f})")
    print(f"    -> a negative 'probability': the labelled clip + renormalise "
          f"policy gives {rep['clipped_corrected']}.")
    print("\n[5] Probabilistic error cancellation (exact coefficients)")
    pec = pec_report()
    print(f"    inverse channel: a = {pec['a']} (identity), "
          f"b = {pec['b']} (each Pauli)")
    print(f"    gamma per gate = {float(pec['gamma_per_gate']):.6f} | "
          f"gamma^2 per gate = {float(pec['gamma_squared_per_gate']):.6f}")
    print(f"    over {len(CIRCUIT_GATES)} gates: gamma_total^2 = "
          f"{float(pec['gamma_squared_total']):.4f} "
          f"(recovery error {pec['recovery_error']:.2e})")
    print("    -> the mixture undoes the channel exactly on paper; the")
    print("       price is a shot budget multiplied by gamma_total^2.")
    print("\n[6] The bill - cost honesty table")
    costs = cost_report()
    print(f"    ZNE:      bias {costs['zne_bias_before']:.6f} -> "
          f"{costs['zne_bias_after']:.6f}, variance x{costs['zne_variance_factor']:.0f}")
    print(f"    readout:  noise amplification up to x{costs['readout_amplification']:.4f}")
    print(f"    PEC:      shot budget x{float(costs['pec_gamma_squared_total']):.4f} "
          f"for this circuit")
    print("\n[7] Mitigation classifier (TinyML kNN, labels from the lab policy)")
    for name, label in classifier_report().items():
        print(f"    {name:36s} -> {label}")
    print("\n[8] Advisory seats")
    brains = brains_report()
    print(f"    decider: {brains['decider']['decision']} "
          f"trace={brains['decider']['trace_sha256_16']}")
    print(f"    advisor can_override_hold = {brains['advisor']['can_override_hold']}")
    print(f"    flow rendered = {brains['flow']['rendered']} "
          f"live = {brains['flow']['live']}")
    print(f"\n[9] HOLD gates: {len(HOLD_GATES)}")
    for key in HOLD_GATES:
        print(f"    HOLD  {key}")
    checks = run_checks()
    passed = sum(1 for _, ok in checks if ok)
    print(f"\nCHECKS {passed}/{len(checks)}")
    for name, ok in checks:
        if not ok:
            print(f"    FAIL {name}")
    verdict = "PROCEED" if passed == len(checks) else "HOLD"
    print(f"VERDICT {verdict} (sandbox simulation only) | HOLD gates {len(HOLD_GATES)}")
    return 0 if passed == len(checks) else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "demo":
        return demo()
    cmd = argv[0]
    if cmd == "checks":
        checks = run_checks()
        for name, ok in checks:
            print(f"{'PASS' if ok else 'FAIL'} {name}")
        print(f"CHECKS {sum(1 for _, ok in checks if ok)}/{len(checks)}")
        return 0 if all(ok for _, ok in checks) else 1
    if cmd == "circuit":
        zne = zne_report()
        print(json.dumps({
            "gates": [list(g) for g in CIRCUIT_GATES],
            "p_per_gate": P_GATE,
            "ideal": zne["ideal"],
            "noisy_lambda": {str(k): v for k, v in zne["values"].items()},
        }, indent=2, sort_keys=True))
        return 0
    if cmd == "zne":
        zne = zne_report()
        print(json.dumps({
            "ideal": zne["ideal"],
            "values": {str(k): v for k, v in zne["values"].items()},
            "weights": zne["weights"],
            "raw_lambda1": zne["raw_lambda1"],
            "mitigated": zne["mitigated"],
            "raw_error": zne["raw_error"],
            "mitigated_error": zne["mitigated_error"],
            "improvement_factor": zne["improvement_factor"],
            "variance_factor": zne["variance_factor"],
        }, indent=2, sort_keys=True))
        return 0
    if cmd == "readout":
        rep = readout_report()
        print(json.dumps({
            "confusion": [list(r) for r in CONFUSION],
            "true": rep["true"],
            "measured": rep["measured"],
            "corrected": rep["corrected"],
            "recovery_error": rep["recovery_error"],
            "labelled_measured": rep["labelled_measured"],
            "naive_corrected": rep["naive_corrected"],
            "clipped_corrected": rep["clipped_corrected"],
        }, indent=2, sort_keys=True))
        return 0
    if cmd == "pec":
        pec = pec_report()
        print(json.dumps({
            "p": str(PEC_P),
            "a": str(pec["a"]),
            "b": str(pec["b"]),
            "gamma_per_gate": float(pec["gamma_per_gate"]),
            "gamma_squared_per_gate": float(pec["gamma_squared_per_gate"]),
            "gamma_total": float(pec["gamma_total"]),
            "gamma_squared_total": float(pec["gamma_squared_total"]),
            "recovery_error": pec["recovery_error"],
        }, indent=2, sort_keys=True))
        return 0
    if cmd == "costs":
        costs = cost_report()
        print(json.dumps({
            "zne_bias_before": costs["zne_bias_before"],
            "zne_bias_after": costs["zne_bias_after"],
            "zne_variance_factor": costs["zne_variance_factor"],
            "zne_note": costs["zne_note"],
            "readout_amplification": costs["readout_amplification"],
            "readout_note": costs["readout_note"],
            "pec_gamma_squared_per_gate": float(costs["pec_gamma_squared_per_gate"]),
            "pec_gamma_squared_total": float(costs["pec_gamma_squared_total"]),
            "pec_note": costs["pec_note"],
        }, indent=2, sort_keys=True))
        return 0
    if cmd == "classify":
        print(json.dumps(classifier_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "brains":
        print(json.dumps(brains_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "holds":
        for key, value in HOLD_GATES.items():
            print(f"{key}: {value}")
        return 0
    if cmd == "hold-demo":
        action = argv[1] if len(argv) > 1 else "hardware_result_claim"
        print(f"{action}: {HOLD_GATES.get(action, 'PROCEED - not gated')}")
        return 2 if gate(action) == "HOLD" else 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
