"""build_operator_lists のベクトル化（O(L^4)→O(L^3)）に対する回帰テスト。

ベクトル化前の素朴な 3 重ループ実装を参照実装として保持し、ベクトル化版の
出力が小規模 L で完全一致することを固定する。あわせて config 検証も確認する。
"""
import contextlib
import io
import unittest

import numpy as np

from config import DEFAULT_CONFIG, deep_merge, prepare_config, validate_config
from simulation import (
    SimulationParams,
    build_bdg_matrix,
    build_operator_lists,
    diagonalize_bdg_matrix,
    enforce_particle_hole_symmetry,
)


def _reference_operator_lists(V, L, PBC):
    """ベクトル化前の素朴な実装（k,l,n の 3 重ループ）。回帰の基準。"""
    cj_dag_cj = []
    for j in range(L):
        M = np.zeros((L, L), dtype=complex)
        for k in range(L):
            for l in range(L):
                M[k, l] = V[j, k].conj() * V[j, l]
                M[k, l] += -1 * V[j, l + L].conj() * V[j, k + L]
                if k == l:
                    for n in range(L):
                        M[k, l] += V[j, n + L].conj() * V[j, n + L]
        cj_dag_cj.append(M)

    def pairing(a, b):
        M = np.zeros((L, L), dtype=complex)
        for k in range(L):
            for l in range(L):
                M[k, l] = V[a, k + L] * V[b, l] - V[a, l] * V[b, k + L]
                if k == l:
                    for n in range(L):
                        M[k, l] += V[a, n] * V[b, n + L]
        return M

    def hopping(a, b):
        M = np.zeros((L, L), dtype=complex)
        for k in range(L):
            for l in range(L):
                M[k, l] = V[a, k].conj() * V[b, l] - V[a, l + L].conj() * V[b, k + L]
                if k == l:
                    for n in range(L):
                        M[k, l] += V[a, n + L].conj() * V[b, n + L]
        return M

    cj1_cj = [pairing(j + 1, j) for j in range(L - 1)]
    cj1_dag_cj = [hopping(j + 1, j) for j in range(L - 1)]
    if PBC:
        cj1_cj.append(pairing(0, L - 1))
        cj1_dag_cj.append(hopping(0, L - 1))
    else:
        cj1_cj.append(np.zeros((L, L), dtype=complex))
        cj1_dag_cj.append(np.zeros((L, L), dtype=complex))
    return cj_dag_cj, cj1_cj, cj1_dag_cj


def _small_eigenvectors(L, PBC):
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "L": L, "l": 2 * np.pi, "p": 0.2, "m": 0.01,
            "scenario": None, "chirality": "chi_plus", "beta_sign": "plus",
            "beta_profile": "pos", "surface_gravity_beta": False, "PBC": PBC,
        },
    )
    with contextlib.redirect_stdout(io.StringIO()):
        config = prepare_config(config)
    params = SimulationParams.from_config(config)
    H = build_bdg_matrix(params)
    _, V = diagonalize_bdg_matrix(H, L)
    return enforce_particle_hole_symmetry(V, L), params


class OperatorVectorizationRegression(unittest.TestCase):
    def test_matches_reference_implementation(self):
        for PBC in (False, True):
            with self.subTest(PBC=PBC):
                L = 6
                V, params = _small_eigenvectors(L, PBC)
                with contextlib.redirect_stdout(io.StringIO()):
                    got = build_operator_lists(V, L, PBC)
                ref = _reference_operator_lists(V, L, PBC)
                for got_list, ref_list in zip(got, ref):
                    self.assertEqual(len(got_list), len(ref_list))
                    for a, b in zip(got_list, ref_list):
                        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-9)


class ConfigValidationTests(unittest.TestCase):
    def _cfg(self, **overrides):
        return deep_merge(DEFAULT_CONFIG, overrides)

    def test_default_config_is_valid(self):
        self.assertIsNotNone(validate_config(self._cfg()))

    def test_rejects_bad_values(self):
        cases = [
            {"L": 1},
            {"L": 10.5},
            {"l": -1},
            {"t_i": 5, "t_f": 1},
            {"PBC": "yes"},
            {"dt_scale": 0},
            {"sigma_fraction": 1.5},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    validate_config(self._cfg(**overrides))


if __name__ == "__main__":
    unittest.main()
