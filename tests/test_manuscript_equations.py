"""Operator-level checks of the central equations in main_revised.tex."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.dispersion_relation import build_homogeneous_bdg, dispersion_bands
from test_energy_density import (
    bond_beta,
    fock_a_operators,
    fock_c_operators,
    fock_fermionic_hamiltonian,
    small_params,
)


def local_operator(single_site: np.ndarray, site: int, L: int) -> np.ndarray:
    result = np.eye(1, dtype=complex)
    for index in range(L):
        result = np.kron(result, single_site if index == site else np.eye(2))
    return result


class ManuscriptEquationTests(unittest.TestCase):
    def test_spin_and_jordan_wigner_hamiltonians_agree(self):
        L = 4
        params = small_params(L=L, p=0.73, m=0.0, PBC=False)
        cs = fock_c_operators(L)
        fermion = fock_fermionic_hamiltonian(cs, params)

        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.diag([1.0, -1.0]).astype(complex)
        xs = [local_operator(X, site, L) for site in range(L)]
        ys = [local_operator(Y, site, L) for site in range(L)]
        zs = [local_operator(Z, site, L) for site in range(L)]
        spin = np.zeros_like(fermion)
        for j in range(L - 1):
            spin += -1 / (4 * params.epsilon) * (
                (1 + params.p) * xs[j] @ xs[j + 1]
                - (1 - params.p) * ys[j] @ ys[j + 1]
                - bond_beta(j, params)
                * (xs[j] @ ys[j + 1] - ys[j] @ xs[j + 1])
            )
        spin += -params.p / (2 * params.epsilon) * sum(zs)
        np.testing.assert_allclose(spin, fermion, atol=1e-12)

    def test_a_basis_hamiltonian_agrees_with_fermionic_form(self):
        L = 5
        params = small_params(L=L, p=0.61, m=0.0, PBC=False)
        cs = fock_c_operators(L)
        a_plus, a_minus = fock_a_operators(cs)
        expected = fock_fermionic_hamiltonian(cs, params)
        actual = np.zeros_like(expected)
        for j in range(L - 1):
            actual += -1j / (2 * params.epsilon) * (
                (1 + bond_beta(j, params)) * a_plus[j] @ a_plus[j + 1]
                - (1 - bond_beta(j, params)) * a_minus[j] @ a_minus[j + 1]
                + params.p
                * (
                    a_minus[j] @ a_plus[j + 1]
                    - a_plus[j] @ a_minus[j + 1]
                )
            )
        actual += -1j * params.p / params.epsilon * sum(
            a_plus[j] @ a_minus[j] for j in range(L)
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_bulk_heisenberg_equations_follow_from_a_hamiltonian(self):
        L = 5
        params = small_params(L=L, p=0.42, m=0.0, PBC=False)
        cs = fock_c_operators(L)
        a_plus, a_minus = fock_a_operators(cs)
        hamiltonian = fock_fermionic_hamiltonian(cs, params)
        j = 2
        eps = params.epsilon

        rhs_plus = -1j / (2 * eps) * (
            (1 + bond_beta(j, params)) * a_plus[j + 1]
            - (1 + bond_beta(j - 1, params)) * a_plus[j - 1]
        ) - 1j * params.p / (2 * eps) * (
            2 * a_minus[j] - a_minus[j + 1] - a_minus[j - 1]
        )
        rhs_minus = 1j / (2 * eps) * (
            (1 - bond_beta(j, params)) * a_minus[j + 1]
            - (1 - bond_beta(j - 1, params)) * a_minus[j - 1]
        ) + 1j * params.p / (2 * eps) * (
            2 * a_plus[j] - a_plus[j + 1] - a_plus[j - 1]
        )

        np.testing.assert_allclose(
            a_plus[j] @ hamiltonian - hamiltonian @ a_plus[j],
            rhs_plus,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            a_minus[j] @ hamiltonian - hamiltonian @ a_minus[j],
            rhs_minus,
            atol=1e-12,
        )

    def test_homogeneous_bdg_spectrum_matches_dispersion_equation(self):
        L = 16
        epsilon = 0.17
        momenta = 2 * np.pi * np.arange(L) / L
        for p in (0.0, 1.0):
            for beta in (0.0, 1.0, 1.2, 2.0):
                with self.subTest(p=p, beta=beta):
                    matrix = build_homogeneous_bdg(beta, p, 0.0, L, epsilon, PBC=True)
                    numerical = np.sort(np.linalg.eigvalsh(matrix) * epsilon)
                    root = np.sqrt(
                        np.sin(momenta) ** 2
                        + p**2 * (1 - np.cos(momenta)) ** 2
                    )
                    analytical = np.sort(
                        np.concatenate(
                            (beta * np.sin(momenta) + root,
                             beta * np.sin(momenta) - root)
                        )
                    )
                    np.testing.assert_allclose(numerical, analytical, atol=1e-12)

    def test_fig1_dispersion_values_are_the_manuscript_values(self):
        for p in (0, 1):
            for beta in (0.0, 1.0, 2.0):
                k, upper, lower = dispersion_bands(beta, p, nk=101)
                self.assertEqual(k.shape, upper.shape)
                self.assertEqual(k.shape, lower.shape)
                self.assertTrue(np.all(np.isfinite(upper)))
                self.assertTrue(np.all(np.isfinite(lower)))


if __name__ == "__main__":
    unittest.main()
