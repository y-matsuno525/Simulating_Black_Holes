import unittest

import numpy as np

from paper_figures.make_fig3a_fft_map import positive_fft
from paper_figures.make_beta12_dispersion import doubler_k
from paper_figures.make_doubler_fft import doubler_k as fig4_doubler_k
from paper_figures.make_fig6b_scaling import DEFAULT_MAX_L, load_scaling_data
from paper_figures.reproduce_panel import (
    FIGURE_GROUPS,
    PANEL_SPECS,
    horizon_positions_from_config,
    normalize_panel_name,
)


class PaperReproductionTests(unittest.TestCase):
    def test_panel_name_normalization(self):
        cases = {
            "FIG2c": "fig2c",
            "FIG2(c)": "fig2c",
            "2c": "fig2c",
            "FIG4b": "fig4b",
            "FIG6(a)": "fig6a",
            "FIG6(b)": "fig6b",
            "FIG7": "fig7",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_panel_name(raw), expected)

    def test_fig2c_manifest_matches_corrected_caption(self):
        spec = PANEL_SPECS["fig2c"]
        cfg = spec.overrides
        self.assertEqual(cfg["L"], 300)
        self.assertEqual(cfg["p"], 0)
        self.assertEqual(cfg["m"], 0.0)
        self.assertEqual(cfg["scenario"], None)
        self.assertEqual(cfg["chirality"], "chi_plus")
        self.assertEqual(cfg["initial_direction"], "right")
        self.assertEqual(cfg["beta_sign"], "plus")
        self.assertAlmostEqual(cfg["beta_amplitude"], 0.6)
        self.assertAlmostEqual(cfg["beta_center_fraction"], 1 / 3)
        self.assertAlmostEqual(cfg["j0_fraction"], 150 / 300)
        self.assertAlmostEqual(cfg["sigma_fraction"], 0.05)
        self.assertEqual(spec.observable, "H_p")
        self.assertAlmostEqual(spec.t_plot, 1.0)

    def test_fig2b_manifest_matches_current_paper_raster(self):
        spec = PANEL_SPECS["fig2b"]
        cfg = spec.overrides
        self.assertEqual(cfg["p"], 0)
        self.assertEqual(cfg["chirality"], "chi_minus")
        self.assertEqual(cfg["initial_direction"], "left")
        self.assertAlmostEqual(cfg["beta_center_fraction"], 2 / 3)
        self.assertAlmostEqual(cfg["j0_fraction"], 120 / 300)
        self.assertEqual(spec.observable, "H_m")

    def test_fig2a_manifest_matches_corrected_caption(self):
        spec = PANEL_SPECS["fig2a"]
        cfg = spec.overrides
        self.assertEqual(cfg["p"], 0)
        self.assertEqual(cfg["chirality"], "chi_plus")
        self.assertEqual(cfg["initial_direction"], "right")
        self.assertAlmostEqual(cfg["beta_center_fraction"], 2 / 3)
        self.assertAlmostEqual(cfg["j0_fraction"], 60 / 300)
        self.assertEqual(spec.observable, "H_p")
        self.assertTrue(spec.show_geodesic)
        self.assertEqual(spec.horizon_color, "black")
        self.assertEqual(spec.horizon_linestyle, ":")
        self.assertEqual(spec.geodesic_color, "cyan")
        self.assertEqual(spec.geodesic_linestyle, "--")
        self.assertTrue(spec.show_legend)

    def test_fig2d_manifest_matches_corrected_caption(self):
        spec = PANEL_SPECS["fig2d"]
        cfg = spec.overrides
        self.assertEqual(cfg["p"], 0)
        self.assertEqual(cfg["chirality"], "chi_minus")
        self.assertEqual(cfg["initial_direction"], "left")
        self.assertAlmostEqual(cfg["beta_center_fraction"], 1 / 3)
        self.assertAlmostEqual(cfg["j0_fraction"], 150 / 300)
        self.assertEqual(spec.observable, "H_m")

    def test_fig3a_fft_helper_returns_positive_k_peak(self):
        sites = np.arange(64)
        mode = 5
        values = np.cos(2 * np.pi * mode * sites / len(sites))[None, :]
        k, spectra = positive_fft(values)
        self.assertEqual(spectra.shape, (1, len(k)))
        self.assertAlmostEqual(float(spectra.max()), 1.0)
        self.assertAlmostEqual(k[int(np.argmax(spectra[0]))], 2 * np.pi * mode / len(sites))

    def test_beta12_p1_doubler_wavenumber(self):
        self.assertAlmostEqual(doubler_k(1.2), 2 * np.arccos(1 / 1.2))
        self.assertAlmostEqual(doubler_k(1.2), 1.171371, places=6)
        self.assertAlmostEqual(fig4_doubler_k(1.2), doubler_k(1.2))

    def test_fig6a_manifest_matches_white_hole_caption(self):
        spec = PANEL_SPECS["fig6a"]
        cfg = spec.overrides
        self.assertEqual(cfg["L"], 300)
        self.assertEqual(cfg["p"], 0)
        self.assertEqual(cfg["scenario"], None)
        self.assertEqual(cfg["chirality"], "chi_plus")
        self.assertEqual(cfg["initial_direction"], "right")
        self.assertEqual(cfg["beta_sign"], "minus")
        self.assertAlmostEqual(cfg["beta_amplitude"], 0.6)
        self.assertAlmostEqual(cfg["beta_center_fraction"], 0.7)
        self.assertAlmostEqual(cfg["j0_fraction"], 0.2)
        self.assertAlmostEqual(cfg["sigma_fraction"], 0.05)
        self.assertEqual(spec.observable, "H_p")
        self.assertAlmostEqual(spec.t_plot, 8.0)
        self.assertTrue(spec.show_geodesic)
        self.assertTrue(spec.show_legend)
        self.assertAlmostEqual(horizon_positions_from_config(spec)[0], 0.743 * 300, delta=0.2)

    def test_fig7_manifest_matches_white_hole_caption(self):
        self.assertEqual(FIGURE_GROUPS["fig7"], ["fig7a", "fig7b", "fig7c"])
        expected = {
            "fig7a": (1, "H_p"),
            "fig7b": (1, "H_m"),
            "fig7c": (1, "H_pm"),
        }
        for key, (p, observable) in expected.items():
            with self.subTest(panel=key):
                spec = PANEL_SPECS[key]
                cfg = spec.overrides
                self.assertEqual(cfg["p"], p)
                self.assertEqual(spec.observable, observable)
                self.assertEqual(cfg["scenario"], None)
                self.assertEqual(cfg["beta_sign"], "minus")
                self.assertAlmostEqual(cfg["beta_center_fraction"], 0.7)
                self.assertAlmostEqual(cfg["j0_fraction"], 0.2)
                self.assertAlmostEqual(spec.t_plot, 8.0)
                self.assertTrue(spec.show_geodesic)
                self.assertTrue(spec.show_legend)
                self.assertEqual(spec.run_id, "FIG7")
                self.assertEqual(cfg["run_name"], "FIG7")
                self.assertEqual(spec.run_dir, PANEL_SPECS["fig7a"].run_dir)

    def test_fig6b_scaling_data_uses_digitized_manuscript_points(self):
        Ls, times, _ = load_scaling_data()
        self.assertEqual(Ls.tolist(), [100, 200, 300, 400, 500, 600, 700, 800])
        self.assertAlmostEqual(float(times[0]), 0.75, places=2)
        self.assertAlmostEqual(float(times[-1]), 4.52, places=2)
        self.assertTrue(np.all(np.isfinite(times)))

    def test_fig6b_default_plot_includes_digitized_l800(self):
        Ls, _, _ = load_scaling_data(max_l=DEFAULT_MAX_L)
        self.assertEqual(float(np.max(Ls)), 800.0)

    def test_spacetime_panels_share_reference_line_style(self):
        keys = ["fig2a", "fig2b", "fig2c", "fig2d", "fig3a", "fig3b", "fig3c", "fig3d", "fig6a", "fig7a", "fig7b", "fig7c"]
        first = PANEL_SPECS[keys[0]]
        for key in keys:
            with self.subTest(panel=key):
                spec = PANEL_SPECS[key]
                self.assertTrue(spec.show_geodesic)
                self.assertTrue(spec.show_legend)
                self.assertEqual(spec.horizon_color, first.horizon_color)
                self.assertEqual(spec.horizon_linestyle, first.horizon_linestyle)
                self.assertEqual(spec.horizon_linewidth, first.horizon_linewidth)
                self.assertEqual(spec.geodesic_color, first.geodesic_color)
                self.assertEqual(spec.geodesic_linestyle, first.geodesic_linestyle)
                self.assertEqual(spec.geodesic_linewidth, first.geodesic_linewidth)


if __name__ == "__main__":
    unittest.main()
