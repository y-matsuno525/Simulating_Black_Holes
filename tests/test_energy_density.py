"""Tests for the revised local energy densities (main_revised.tex Eqs. (40)-(43)).

The local densities are verified against an independent full-Fock-space
construction: Jordan-Wigner fermions on a small chain, the a^+/a^- operators
of Eq. (38), and the site densities of Eqs. (41)-(43) built literally from
those operators.  The quasiparticle-basis matrices produced by
``build_energy_densities()`` and the scalars from ``compute_vacuum_values()``
must reproduce the exact Fock-space matrix elements, and
``epsilon * sum_j H_j`` must equal the fermionic Hamiltonian including the
known onsite constant.
"""

import contextlib
import io
import unittest

import numpy as np

from config import DEFAULT_CONFIG, beta_from_config, deep_merge, prepare_config
from simulation import (
    SimulationParams,
    build_bdg_matrix,
    build_energy_densities,
    build_operator_lists,
    compute_std,
    compute_vacuum_values,
    diagonalize_bdg_matrix,
    enforce_particle_hole_symmetry,
    is_hermitian,
)


def prepared_config(overrides):
    with contextlib.redirect_stdout(io.StringIO()):
        return prepare_config(deep_merge(DEFAULT_CONFIG, overrides))


def small_params(**overrides):
    config = prepared_config(
        deep_merge(
            {
                "L": 6,
                "l": 2 * np.pi,
                "p": 1.0,
                "m": 0.0,
                "scenario": None,
                "chirality": "chi_plus",
                "beta_sign": "plus",
                "beta_profile": "pos",
                "beta_amplitude": 0.6,
                "beta_width": 1.0,
                "beta_center_fraction": 0.5,
                "surface_gravity_beta": False,
                "PBC": False,
                "mode_function_count": 0,
                "outputs": {
                    "show_beta_profile": False,
                    "heatmaps": False,
                    "gifs": False,
                    "mode_functions": False,
                    "geodesic": False,
                    "surface_gravity_fit": False,
                    "fft": False,
                },
            },
            overrides,
        )
    )
    return SimulationParams.from_config(config)


def bond_beta(b, params):
    """Beta at the center of bond (b, b+1); the PBC wrap bond is b = L-1."""
    return beta_from_config((b % params.L) + 0.5, params.config)


def fock_c_operators(L):
    """Jordan-Wigner fermion annihilation operators as 2^L x 2^L matrices."""
    annihilate = np.array([[0.0, 1.0], [0.0, 0.0]])
    Z = np.diag([1.0, -1.0])
    I2 = np.eye(2)
    cs = []
    for j in range(L):
        op = np.eye(1)
        for k in range(L):
            if k < j:
                op = np.kron(op, Z)
            elif k == j:
                op = np.kron(op, annihilate)
            else:
                op = np.kron(op, I2)
        cs.append(op)
    return cs


def fock_a_operators(cs):
    """Hermitian a^+/a^- operators from Eq. (38) of main_revised.tex."""
    a_plus = [0.5 * ((1 + 1j) * c + (1 - 1j) * c.conj().T) for c in cs]
    a_minus = [0.5 * ((-1 + 1j) * c - (1 + 1j) * c.conj().T) for c in cs]
    return a_plus, a_minus


def existing_bonds(params):
    """List of bond indices b for bonds (b, (b+1) mod L) present in the chain."""
    bonds = list(range(params.L - 1))
    if params.PBC:
        bonds.append(params.L - 1)
    return bonds


def paper_fock_densities(cs, params):
    """Site densities of Eqs. (41)-(43) built literally in Fock space.

    The onsite c^dag c coefficient uses (p - epsilon*m) to match the code's
    Hamiltonian; for m = 0 this is exactly Eq. (43).
    """
    L = params.L
    eps = params.epsilon
    p = params.p
    dim = cs[0].shape[0]
    a_p, a_m = fock_a_operators(cs)
    bonds = set(existing_bonds(params))

    def right_site(b):
        return (b + 1) % L

    H_plus = []
    H_minus = []
    H_int = []
    onsite_coeff = p - eps * params.m
    for j in range(L):
        left = (j - 1) % L
        right = j
        Hp = np.zeros((dim, dim), dtype=complex)
        Hm = np.zeros((dim, dim), dtype=complex)
        Hi = np.zeros((dim, dim), dtype=complex)
        # bond (j, j+1): a_j a_{j+1} ordering; bond (j-1, j): a_{j-1} a_j.
        if right in bonds:
            r = right_site(right)
            Hp += -1j / (4 * eps**2) * (1 + bond_beta(right, params)) * (a_p[j] @ a_p[r])
            Hm += 1j / (4 * eps**2) * (1 - bond_beta(right, params)) * (a_m[j] @ a_m[r])
            Hi += -1j * p / (4 * eps**2) * (a_m[j] @ a_p[r] - a_p[j] @ a_m[r])
        if left in bonds:
            lsite = left
            Hp += -1j / (4 * eps**2) * (1 + bond_beta(left, params)) * (a_p[lsite] @ a_p[j])
            Hm += 1j / (4 * eps**2) * (1 - bond_beta(left, params)) * (a_m[lsite] @ a_m[j])
            Hi += -1j * p / (4 * eps**2) * (a_m[lsite] @ a_p[j] - a_p[lsite] @ a_m[j])
        Hi += -1j * onsite_coeff / (eps**2) * (a_p[j] @ a_m[j])
        H_plus.append(Hp)
        H_minus.append(Hm)
        H_int.append(Hi)
    return H_plus, H_minus, H_int


def fock_fermionic_hamiltonian(cs, params):
    """Lattice Hamiltonian of Eq. (30) with the code's (p - eps*m) onsite term."""
    L = params.L
    eps = params.epsilon
    p = params.p
    dim = cs[0].shape[0]
    H = np.zeros((dim, dim), dtype=complex)
    for b in existing_bonds(params):
        b1 = (b + 1) % L
        beta_link = bond_beta(b, params)
        H += -1 / (2 * eps) * (
            cs[b1] @ cs[b]
            + cs[b].conj().T @ cs[b1].conj().T
            + p * (cs[b].conj().T @ cs[b1] + cs[b1].conj().T @ cs[b])
            + 1j * beta_link * (cs[b].conj().T @ cs[b1] - cs[b1].conj().T @ cs[b])
        )
    onsite_coeff = p - eps * params.m
    for j in range(L):
        H += -onsite_coeff / (2 * eps) * (np.eye(dim) - 2 * cs[j].conj().T @ cs[j])
    return H


def fock_quasiparticle_basis(cs, eigenvectors, L):
    """Return (vacuum, [gamma_n^dag |vac>]) for the code's BdG eigenbasis."""
    gammas = []
    for n in range(L):
        gamma = np.zeros(cs[0].shape, dtype=complex)
        for j in range(L):
            gamma += eigenvectors[j, n].conj() * cs[j]
            gamma += eigenvectors[j + L, n].conj() * cs[j].conj().T
        gammas.append(gamma)
    number_op = sum(g.conj().T @ g for g in gammas)
    vals, vecs = np.linalg.eigh(number_op)
    assert vals[0] < 1e-8, "no Fock state annihilated by all gamma_n"
    assert vals[1] > 1e-6, "quasiparticle vacuum is not unique"
    vac = vecs[:, 0]
    basis = [g.conj().T @ vac for g in gammas]
    return vac, basis


def code_density_matrices(params):
    H_BdG = build_bdg_matrix(params)
    eigenvalues, eigenvectors = diagonalize_bdg_matrix(H_BdG, params.L)
    eigenvectors = enforce_particle_hole_symmetry(eigenvectors, params.L)
    with contextlib.redirect_stdout(io.StringIO()):
        op_lists = build_operator_lists(eigenvectors, params.L, params.PBC)
        densities = build_energy_densities(*op_lists, params)
        vacuum = compute_vacuum_values(eigenvectors, params)
    return eigenvalues, eigenvectors, densities, vacuum


class EnergyDensityFockTests(unittest.TestCase):
    """Compare the coded densities with a literal Fock-space construction."""

    def assert_matches_fock(self, params):
        L = params.L
        eigenvalues, eigenvectors, densities, vacuum = code_density_matrices(params)
        cs = fock_c_operators(L)
        Hp_fock, Hm_fock, Hi_fock = paper_fock_densities(cs, params)
        vac, basis = fock_quasiparticle_basis(cs, eigenvectors, L)

        for key, fock_ops in (("H_p", Hp_fock), ("H_m", Hm_fock), ("H_pm", Hi_fock)):
            for j in range(L):
                expected = np.empty((L, L), dtype=complex)
                for k in range(L):
                    for l in range(L):
                        expected[k, l] = basis[k].conj() @ fock_ops[j] @ basis[l]
                np.testing.assert_allclose(
                    densities[key][j], expected, atol=1e-10,
                    err_msg=f"{key}[{j}] differs from the Fock-space Eq. (41)-(43) operator",
                )
                vac_expected = vac.conj() @ fock_ops[j] @ vac
                np.testing.assert_allclose(
                    vacuum[key][j], vac_expected, atol=1e-10,
                    err_msg=f"vacuum {key}[{j}] differs from the Fock-space value",
                )

    def test_open_boundary_matches_fock_construction(self):
        self.assert_matches_fock(small_params(PBC=False))

    def test_periodic_boundary_matches_fock_construction(self):
        self.assert_matches_fock(small_params(PBC=True))

    def test_p_zero_matches_fock_construction(self):
        self.assert_matches_fock(small_params(p=0.0))

    def test_sum_rule_epsilon_sum_equals_hamiltonian(self):
        # Nonzero m pins the explicitly known onsite constant -(p - eps*m)/(2 eps^2).
        for PBC in (False, True):
            with self.subTest(PBC=PBC):
                params = small_params(PBC=PBC, m=0.01)
                cs = fock_c_operators(params.L)
                Hp_fock, Hm_fock, Hi_fock = paper_fock_densities(cs, params)
                total = params.epsilon * sum(
                    Hp_fock[j] + Hm_fock[j] + Hi_fock[j] for j in range(params.L)
                )
                H_ferm = fock_fermionic_hamiltonian(cs, params)
                np.testing.assert_allclose(total, H_ferm, atol=1e-10)

    def test_sum_rule_in_quasiparticle_sector(self):
        # epsilon * sum_j H_j must act as diag(E_n) + E_vac in the
        # one-quasiparticle sector, with E_vac = epsilon * sum_j (vacuum values).
        for PBC in (False, True):
            with self.subTest(PBC=PBC):
                params = small_params(PBC=PBC, m=0.01)
                L = params.L
                eigenvalues, _, densities, vacuum = code_density_matrices(params)
                total = params.epsilon * sum(
                    densities["H_p"][j] + densities["H_m"][j] + densities["H_pm"][j]
                    for j in range(L)
                )
                E_vac = params.epsilon * sum(
                    vacuum["H_p"][j] + vacuum["H_m"][j] + vacuum["H_pm"][j]
                    for j in range(L)
                )
                expected = np.diag(eigenvalues[:L]) + E_vac * np.eye(L)
                np.testing.assert_allclose(total, expected, atol=1e-8)


class EnergyDensityStructureTests(unittest.TestCase):
    def test_local_densities_are_hermitian(self):
        for PBC in (False, True):
            with self.subTest(PBC=PBC):
                params = small_params(PBC=PBC)
                _, _, densities, _ = code_density_matrices(params)
                for key in ("H_p", "H_m", "H_pm"):
                    for j, op in enumerate(densities[key]):
                        self.assertTrue(is_hermitian(op), f"{key}[{j}] not Hermitian")

    def test_open_boundary_endpoints_keep_one_sided_bond_halves(self):
        params = small_params(PBC=False)
        L = params.L
        _, _, densities, _ = code_density_matrices(params)
        for key in ("H_p", "H_m"):
            first = densities[key][0]
            last = densities[key][L - 1]
            self.assertGreater(np.abs(first).max(), 0.0, f"{key}[0] must not be zeroed")
            self.assertGreater(np.abs(last).max(), 0.0, f"{key}[L-1] must not be zeroed")

    def test_left_and_right_links_use_their_own_beta(self):
        # Two profiles that agree on the right link of site j but differ on the
        # left link must give different H_j; averaging a single beta cannot.
        params = small_params(PBC=False)
        L = params.L
        eps = params.epsilon
        cs = fock_c_operators(L)
        a_p, _ = fock_a_operators(cs)
        j = 3
        beta_left = bond_beta(j - 1, params)
        beta_right = bond_beta(j, params)
        self.assertNotAlmostEqual(beta_left, beta_right)

        _, eigenvectors, densities, _ = code_density_matrices(params)
        vac, basis = fock_quasiparticle_basis(cs, eigenvectors, L)
        left_bond = a_p[j - 1] @ a_p[j]
        right_bond = a_p[j] @ a_p[j + 1]
        expected_op = -1j / (4 * eps**2) * (
            (1 + beta_left) * left_bond + (1 + beta_right) * right_bond
        )
        expected = np.array(
            [[basis[k].conj() @ expected_op @ basis[l] for l in range(L)] for k in range(L)]
        )
        np.testing.assert_allclose(densities["H_p"][j], expected, atol=1e-10)

        # A single averaged beta on both links (the old behaviour) must fail.
        wrong_op = -1j / (4 * eps**2) * (1 + beta_right) * (left_bond + right_bond)
        wrong = np.array(
            [[basis[k].conj() @ wrong_op @ basis[l] for l in range(L)] for k in range(L)]
        )
        self.assertGreater(np.abs(densities["H_p"][j] - wrong).max(), 1e-6)


class ProfileMeasurementTests(unittest.TestCase):
    def test_positive_rescaling_keeps_center_and_width(self):
        rng = np.random.default_rng(7)
        profile = np.abs(rng.normal(size=40)) + 0.1
        x = np.arange(len(profile))
        center = np.sum(profile / profile.sum() * x)
        std = compute_std(profile)
        for scale in (2.5, 1e-3, 1e4):
            scaled = scale * profile
            scaled_center = np.sum(scaled / scaled.sum() * x)
            self.assertAlmostEqual(center, scaled_center, places=12)
            self.assertAlmostEqual(std, compute_std(scaled), places=12)


if __name__ == "__main__":
    unittest.main()
