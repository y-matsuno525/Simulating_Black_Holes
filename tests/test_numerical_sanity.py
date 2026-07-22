import contextlib
import io
import unittest

import numpy as np

from config import (
    DEFAULT_CONFIG,
    beta_from_config,
    compute_horizon_positions_from_config,
    deep_merge,
    prepare_config,
)
from simulation import (
    SimulationParams,
    build_bdg_matrix,
    build_energy_densities,
    build_operator_lists,
    compute_vacuum_values,
    diagonalize_bdg_matrix,
    enforce_particle_hole_symmetry,
    is_hermitian,
)


def prepared_config(overrides):
    """Prepare a small deterministic config without noisy override prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return prepare_config(deep_merge(DEFAULT_CONFIG, overrides))


class ConfigSanityTests(unittest.TestCase):
    def test_surface_gravity_overrides_are_recorded(self):
        config = prepared_config(
            {
                "L": 12,
                "surface_gravity_beta": True,
                "chirality": "chi_plus",
                "beta_sign": "minus",
                "j0_fraction": 0.2,
                "sigma_fraction": 0.05,
            }
        )

        self.assertEqual(config["scenario"], "BH_chi_minus")
        self.assertEqual(config["chirality"], "chi_minus")
        self.assertEqual(config["beta_sign"], "plus")
        self.assertEqual(config["j0_fraction"], 0.48)
        self.assertEqual(config["sigma_fraction"], 0.003)
        overridden_keys = {note["key"] for note in config["config_override_notes"]}
        self.assertGreaterEqual(
            overridden_keys,
            {"scenario", "chirality", "beta_sign", "j0_fraction", "sigma_fraction"},
        )

    def test_pos_horizon_position_matches_analytic_formula(self):
        config = prepared_config(
            {
                "L": 60,
                "l": 2*np.pi,
                "scenario": None,
                "chirality": "chi_plus",
                "beta_sign": "plus",
                "beta_profile": "pos",
                "beta_amplitude": 0.6,
                "beta_width": 1.0,
                "beta_center_fraction": 2/3,
                "surface_gravity_beta": False,
            }
        )

        horizon = compute_horizon_positions_from_config(config, num_samples=20000)[0]
        expected = (
            config["beta_center_fraction"] * config["L"]
            + config["beta_width"] * np.arctanh(1/config["beta_amplitude"] - 1)
            / (3 * config["epsilon"])
        )
        self.assertAlmostEqual(horizon, expected, places=3)


class NumericalSanityTests(unittest.TestCase):
    def small_params(self, **overrides):
        config = prepared_config(
            deep_merge(
                {
                    "L": 8,
                    "l": 2*np.pi,
                    "p": 0.2,
                    "m": 0.01,
                    "scenario": None,
                    "chirality": "chi_plus",
                    "beta_sign": "plus",
                    "beta_profile": "flat",
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

    def test_bdg_matrix_is_hermitian_for_open_and_periodic_boundaries(self):
        for PBC in (False, True):
            with self.subTest(PBC=PBC):
                params = self.small_params(PBC=PBC)
                H_BdG = build_bdg_matrix(params)
                self.assertTrue(is_hermitian(H_BdG))

    def test_bdg_beta_terms_use_link_center_for_nonuniform_profile(self):
        params = self.small_params(
            L=8,
            p=0.3,
            beta_profile="pos",
            beta_amplitude=0.6,
            beta_width=1.0,
            beta_center_fraction=0.5,
            PBC=False,
        )
        H_BdG = build_bdg_matrix(params)
        a = 4
        beta_link = beta_from_config(a + 0.5, params.config)
        pref = -1/(2*params.epsilon)
        L = params.L

        np.testing.assert_allclose(H_BdG[a+1, a], pref * (params.p - 1j*beta_link))
        np.testing.assert_allclose(H_BdG[a, a+1], pref * (params.p + 1j*beta_link))
        np.testing.assert_allclose(H_BdG[L+a+1, L+a], pref * (-params.p - 1j*beta_link))
        np.testing.assert_allclose(H_BdG[L+a, L+a+1], pref * (-params.p + 1j*beta_link))

    def test_bdg_periodic_boundary_uses_last_link_center_beta(self):
        params = self.small_params(
            L=8,
            p=0.3,
            beta_profile="pos",
            beta_amplitude=0.6,
            beta_width=1.0,
            beta_center_fraction=0.5,
            PBC=True,
        )
        H_BdG = build_bdg_matrix(params)
        L = params.L
        beta_boundary = beta_from_config(L - 0.5, params.config)
        pref = -1/(2*params.epsilon)

        np.testing.assert_allclose(H_BdG[0, L-1], pref * (params.p - 1j*beta_boundary))
        np.testing.assert_allclose(H_BdG[L-1, 0], pref * (params.p + 1j*beta_boundary))
        np.testing.assert_allclose(H_BdG[L, 2*L-1], pref * (-params.p - 1j*beta_boundary))
        np.testing.assert_allclose(H_BdG[2*L-1, L], pref * (-params.p + 1j*beta_boundary))

    def test_local_energy_densities_and_vacuum_values_are_hermitian(self):
        params = self.small_params(L=6)
        H_BdG = build_bdg_matrix(params)
        _, eigenvectors = diagonalize_bdg_matrix(H_BdG, params.L)
        eigenvectors = enforce_particle_hole_symmetry(eigenvectors, params.L)
        with contextlib.redirect_stdout(io.StringIO()):
            cj_dag_cj, cj1_cj, cj1_dag_cj = build_operator_lists(
                eigenvectors,
                params.L,
                params.PBC,
            )

        energy_densities = build_energy_densities(cj_dag_cj, cj1_cj, cj1_dag_cj, params)
        for name, operators in energy_densities.items():
            for j, operator in enumerate(operators):
                with self.subTest(observable=name, site=j):
                    self.assertTrue(is_hermitian(operator))

        vacuum_values = compute_vacuum_values(eigenvectors, params)
        for name in ("H_p", "H_m", "H_pm"):
            max_imag = np.max(np.abs(np.imag(vacuum_values[name])))
            self.assertLess(max_imag, 1e-8)


if __name__ == "__main__":
    unittest.main()
