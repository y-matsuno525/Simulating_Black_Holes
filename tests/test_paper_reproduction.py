import unittest

import numpy as np

from paper_figures.make_fig3a_fft_map import positive_fft
from paper_figures.make_beta12_dispersion import doubler_k
from paper_figures.make_dispersion import BETAS as FIG1_BETAS
from paper_figures.make_doubler_fft import FFT_TIMES, doubler_k as fig4_doubler_k, fft_snapshots
from paper_figures.make_fig6b_scaling import DEFAULT_MAX_L, load_scaling_data
from paper_figures.make_new_layout import (
    LAYOUTS,
    PROFILE_CUT_TIMES,
    PROFILE_CUT_TIMES_BY_KEY,
    _panel_specs,
    _raw_scale,
    load_profile_cut_data,
    profile_cut_times,
    profile_cut_y_limit,
)
from paper_figures.make_surface_gravity import SG_SETTINGS, _build_sg_config, lattice_width_to_physical
from paper_figures.measure_fig6b_scaling import (
    CURVATURE_THRESHOLD,
    WH_X0,
    beta_second_derivative,
    build_config as build_fig6b_config,
    central_curvature_boundary,
    load_results as load_measured_fig6b_results,
)
from paper_figures.reproduce_panel import (
    FIGURE_GROUPS,
    NEW_LAYOUT_TARGETS,
    PANEL_SPECS,
    PROFILE_CUT_TARGETS,
    horizon_positions_from_config,
    load_panel_plot_data,
    load_time_site_csv,
    normalize_panel_name,
)


class PaperReproductionTests(unittest.TestCase):
    def test_fig1_columns_match_caption(self):
        self.assertEqual(FIG1_BETAS, (0.0, 1.0, 2.0))

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

    def test_meaning_based_new_layout_targets_are_available(self):
        self.assertEqual(
            NEW_LAYOUT_TARGETS,
            {
                "bhp0aplus",
                "bhp0aminus",
                "bhp1aplus",
                "bhp1aminus",
                "newlayout",
            },
        )
        self.assertEqual(normalize_panel_name("BH_P0_APLUS"), "bhp0aplus")
        self.assertEqual(normalize_panel_name("NEW_LAYOUT"), "newlayout")

    def test_profile_cut_targets_are_available(self):
        self.assertEqual(
            PROFILE_CUT_TARGETS,
            {
                "bhprofilecuts": None,
                "bhp0apluscuts": "bhp0aplus",
                "bhp0aminuscuts": "bhp0aminus",
                "bhp1apluscuts": "bhp1aplus",
                "bhp1aminuscuts": "bhp1aminus",
            },
        )
        self.assertEqual(normalize_panel_name("BH_PROFILE_CUTS"), "bhprofilecuts")

    def test_profile_cuts_use_exact_unmodified_csv_rows(self):
        self.assertEqual(PROFILE_CUT_TIMES, (2.0, 3.5))
        self.assertEqual(
            PROFILE_CUT_TIMES_BY_KEY,
            {
                "bhp0aplus": (2.0, 3.5),
                "bhp0aminus": (0.5, 0.9),
                "bhp1aplus": (2.0, 3.5),
                "bhp1aminus": (0.5, 0.9),
            },
        )
        for key in LAYOUTS:
            definition = LAYOUTS[key]
            specs, arrays, indices, cuts = load_profile_cut_data(definition)
            for cut_time in profile_cut_times(definition):
                with self.subTest(target=definition.key, time=cut_time):
                    row = indices[cut_time]
                    source_times, _ = load_time_site_csv(
                        specs[0].run_dir / "H_p_val.csv"
                    )
                    self.assertEqual(source_times[row], cut_time)
                    for observable in ("H_p", "H_m", "H_pm"):
                        np.testing.assert_array_equal(
                            cuts[cut_time][observable], arrays[observable][row]
                        )

    def test_profile_cut_ylim_is_shared_and_symmetric(self):
        for key in LAYOUTS:
            definition = LAYOUTS[key]
            _, _, _, cuts = load_profile_cut_data(definition)
            limit = profile_cut_y_limit(cuts)
            direct = max(
                float(np.max(np.abs(values)))
                for cut in cuts.values()
                for values in cut.values()
            )
            self.assertAlmostEqual(limit, 1.05 * direct)

    def test_p_zero_profile_cut_suppressed_components_are_unmodified(self):
        _, arrays, _, _ = load_profile_cut_data(LAYOUTS["bhp0aplus"])
        self.assertLessEqual(float(np.max(np.abs(arrays["H_m"]))), 1.1e-12)
        self.assertTrue(np.array_equal(arrays["H_pm"], np.zeros_like(arrays["H_pm"])))

        specs = _panel_specs(LAYOUTS["bhp0aminus"])
        _, h_plus = load_time_site_csv(specs[0].run_dir / "H_p_val.csv")
        _, h_int = load_time_site_csv(specs[0].run_dir / "H_pm_val.csv")
        self.assertLessEqual(float(np.max(np.abs(h_plus))), 1.1e-12)
        self.assertTrue(np.array_equal(h_int, np.zeros_like(h_int)))

    def test_a_minus_profile_runs_use_available_early_time_rows(self):
        for key in ("bhp0aminus", "bhp1aminus"):
            with self.subTest(target=key):
                definition = LAYOUTS[key]
                specs, arrays, indices, cuts = load_profile_cut_data(definition)
                self.assertEqual(tuple(cuts), (0.5, 0.9))
                for cut_time in profile_cut_times(definition):
                    row = indices[cut_time]
                    for observable in ("H_p", "H_m", "H_pm"):
                        np.testing.assert_array_equal(
                            cuts[cut_time][observable], arrays[observable][row]
                        )

    def test_new_layout_reuses_four_exterior_runs_for_all_components(self):
        expected = {
            "bhp0aplus": ("FIG2a", 0, "chi_plus", 60 / 300),
            "bhp0aminus": ("FIG2b", 0, "chi_minus", 120 / 300),
            "bhp1aplus": ("FIG3a", 1, "chi_plus", 60 / 300),
            "bhp1aminus": ("FIG3b", 1, "chi_minus", 120 / 300),
        }
        for key, (run_id, p, chirality, j0_fraction) in expected.items():
            with self.subTest(target=key):
                specs = _panel_specs(LAYOUTS[key])
                self.assertEqual([spec.observable for spec in specs], ["H_p", "H_m", "H_pm"])
                self.assertEqual([spec.tag for spec in specs], ["a", "b", "c"])
                self.assertEqual({spec.run_dir.name for spec in specs}, {run_id})
                self.assertEqual({spec.overrides["p"] for spec in specs}, {p})
                self.assertEqual({spec.overrides["chirality"] for spec in specs}, {chirality})
                self.assertEqual({spec.overrides["j0_fraction"] for spec in specs}, {j0_fraction})

    def test_new_layout_loads_stored_csv_values_without_rescaling(self):
        for definition in LAYOUTS.values():
            for spec in _panel_specs(definition):
                with self.subTest(target=definition.key, observable=spec.observable):
                    source_times, source_values = load_time_site_csv(
                        spec.run_dir / f"{spec.observable}_val.csv"
                    )
                    times, values, _, _, _, _ = load_panel_plot_data(spec)
                    keep = source_times <= spec.t_plot
                    np.testing.assert_array_equal(times, source_times[keep])
                    np.testing.assert_array_equal(values, source_values[keep])

    def test_new_layout_scale_override_changes_only_color_limits(self):
        spec = _panel_specs(LAYOUTS["bhp0aplus"])[1]
        _, baseline, _, _, _, _ = load_panel_plot_data(spec)
        _, overridden, _, norm, _, _ = load_panel_plot_data(spec, scale_override=7.5)
        np.testing.assert_array_equal(overridden, baseline)
        self.assertAlmostEqual(norm.vmin, -7.5)
        self.assertAlmostEqual(norm.vmax, 7.5)

    def test_p0_common_scale_is_computed_from_unmodified_components(self):
        for key in ("bhp0aplus", "bhp0aminus"):
            with self.subTest(target=key):
                specs = _panel_specs(LAYOUTS[key])
                scales = [_raw_scale(spec) for spec in specs]
                direct = []
                for spec in specs:
                    _, values = load_time_site_csv(spec.run_dir / f"{spec.observable}_val.csv")
                    direct.append(float(np.max(np.abs(values))))
                np.testing.assert_allclose(scales, direct, rtol=0.0, atol=0.0)

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

    def test_fig3_manifests_match_fig2_caption_with_p_one(self):
        for tag in "abcd":
            with self.subTest(panel=tag):
                fig2 = PANEL_SPECS[f"fig2{tag}"]
                fig3 = PANEL_SPECS[f"fig3{tag}"]
                self.assertEqual(fig3.overrides["p"], 1)
                for key in (
                    "L", "l", "m", "chirality", "initial_direction",
                    "beta_sign", "beta_amplitude", "beta_width",
                    "beta_center_fraction", "j0_fraction", "sigma_fraction",
                ):
                    self.assertEqual(fig3.overrides[key], fig2.overrides[key])
                self.assertEqual(fig3.observable, fig2.observable)
                self.assertEqual(fig3.t_plot, fig2.t_plot)

    def test_beta12_p1_doubler_wavenumber(self):
        self.assertAlmostEqual(doubler_k(1.2), 2 * np.arccos(1 / 1.2))
        self.assertAlmostEqual(doubler_k(1.2), 1.171371, places=6)
        self.assertAlmostEqual(fig4_doubler_k(1.2), doubler_k(1.2))

    def test_fig4_fft_requires_profiles_at_the_four_caption_times(self):
        data = np.zeros((len(FFT_TIMES), 16))
        k, spectra = fft_snapshots(data)
        self.assertEqual(spectra.shape, (4, len(k)))
        with self.assertRaises(ValueError):
            fft_snapshots(data[:3])

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
        self.assertAlmostEqual(cfg["beta_center_fraction"], 2 / 3)
        self.assertAlmostEqual(cfg["j0_fraction"], 0.2)
        self.assertAlmostEqual(cfg["sigma_fraction"], 0.05)
        self.assertEqual(spec.observable, "H_p")
        self.assertAlmostEqual(spec.t_plot, 8.0)
        self.assertTrue(spec.show_geodesic)
        self.assertTrue(spec.show_legend)
        # main_revised.tex: x0 = 2*ell/3 puts the white-hole horizon at j_h ~ 213.
        self.assertAlmostEqual(horizon_positions_from_config(spec)[0], 212.81, delta=0.2)

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
                self.assertAlmostEqual(cfg["beta_center_fraction"], 2 / 3)
                self.assertAlmostEqual(cfg["j0_fraction"], 0.2)
                self.assertAlmostEqual(spec.t_plot, 8.0)
                self.assertTrue(spec.show_geodesic)
                self.assertTrue(spec.show_legend)
                self.assertEqual(spec.run_id, "FIG7")
                self.assertEqual(cfg["run_name"], "FIG7")
                self.assertEqual(spec.run_dir, PANEL_SPECS["fig7a"].run_dir)

    def test_spacetime_panel_labels_match_original_manuscript_notation(self):
        expected = {
            "fig2a": r"$\mathcal{H}_{j}^{+}$",
            "fig2b": r"$\mathcal{H}_{j}^{-}$",
            "fig3a": r"$\mathcal{H}_{j}^{+}$",
            "fig3b": r"$\mathcal{H}_{j}^{-}$",
            "fig6a": r"$\mathcal{H}_{j}^{+}$",
            "fig7a": r"$\mathcal{H}_{j}^{+}$",
            "fig7b": r"$\mathcal{H}_{j}^{-}$",
            "fig7c": r"$\mathcal{H}_{j}^{\mathrm{int}}$",
        }
        for key, label in expected.items():
            with self.subTest(panel=key):
                self.assertEqual(PANEL_SPECS[key].label, label)

    def test_fig6b_scaling_data_uses_measured_points(self):
        Ls, times, _ = load_scaling_data()
        self.assertEqual(Ls.tolist(), [100, 200, 300, 400, 500, 600, 700, 800])
        self.assertTrue(np.all(np.isfinite(times)))

    def test_fig6b_default_plot_includes_measured_l800(self):
        Ls, _, _ = load_scaling_data(max_l=DEFAULT_MAX_L)
        self.assertEqual(float(np.max(Ls)), 800.0)

    def test_fig6b_measured_data_are_finite(self):
        Ls, _, times = load_measured_fig6b_results()
        self.assertEqual(Ls.tolist(), [100, 200, 300, 400, 500, 600, 700, 800])
        self.assertTrue(np.all(np.isfinite(times)))

    def test_fig6b_curvature_boundary_matches_manuscript_threshold(self):
        boundary = central_curvature_boundary()
        self.assertLess(boundary, WH_X0)
        self.assertAlmostEqual(
            abs(float(beta_second_derivative(np.asarray(boundary)))),
            CURVATURE_THRESHOLD,
            places=10,
        )

    def test_surface_gravity_width_uses_ell_over_L_once(self):
        lattice_width = np.array([0.0, 1.0, 2.5])
        physical_width = lattice_width_to_physical(lattice_width, ell=2 * np.pi, L=500)
        np.testing.assert_allclose(physical_width, lattice_width * (2 * np.pi / 500))

    def test_fig5_manifest_matches_caption(self):
        self.assertEqual(SG_SETTINGS["L"], 500)
        self.assertAlmostEqual(SG_SETTINGS["l"], 2 * np.pi)
        self.assertEqual(SG_SETTINGS["chirality"], "chi_minus")
        self.assertEqual(SG_SETTINGS["initial_direction"], "left")
        self.assertEqual(SG_SETTINGS["beta_profile"], "center")
        self.assertEqual(SG_SETTINGS["centered_beta_amplitude"], 1.0)
        self.assertEqual(SG_SETTINGS["centered_beta_width"], 1.0)
        self.assertEqual(SG_SETTINGS["centered_beta_center_fraction"], 0.5)
        self.assertEqual(SG_SETTINGS["j0_fraction"], 245 / 500)
        self.assertEqual(SG_SETTINGS["sigma_fraction"], 0.003)
        for p in (0, 1):
            config = _build_sg_config(p)
            self.assertEqual(config["p"], p)
            self.assertEqual(config["j0"], 245)
            self.assertAlmostEqual(config["sigma"], 1.5)

    def test_fig6b_all_sizes_use_caption_settings(self):
        for L in (100, 300, 500, 800):
            with self.subTest(L=L):
                config = build_fig6b_config(L)
                self.assertEqual(config["L"], L)
                self.assertEqual(config["p"], 0)
                self.assertEqual(config["beta_sign"], "minus")
                self.assertAlmostEqual(config["beta_amplitude"], 0.6)
                self.assertAlmostEqual(config["beta_center_fraction"], 2 / 3)
                self.assertEqual(config["j0"], int(0.2 * L))
                self.assertAlmostEqual(config["sigma"], 0.05 * L)

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
                self.assertEqual(spec.geodesic_linestyle, first.geodesic_linestyle)

    def test_faint_trajectory_panels_use_asinh_and_emphasized_geodesics(self):
        expected_widths = {"fig3a": 1.0, "fig6a": 3.0}
        for key, expected_width in expected_widths.items():
            with self.subTest(panel=key):
                spec = PANEL_SPECS[key]
                self.assertEqual(spec.color_norm, "asinh")
                self.assertEqual(spec.asinh_linear_width, expected_width)
                self.assertEqual(spec.geodesic_color, "#009FB7")
                self.assertEqual(spec.geodesic_linewidth, 1.4)

        for key in ("fig3b", "fig3c", "fig3d", "fig7a", "fig7b", "fig7c"):
            with self.subTest(panel=key):
                self.assertEqual(PANEL_SPECS[key].color_norm, "linear")


if __name__ == "__main__":
    unittest.main()
