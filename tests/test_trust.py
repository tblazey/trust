import json
import os

import nibabel as nib
import numpy as np
import pytest

from trust import ExpModel, make_roi, r2_to_oxy
from trust.cli import main

# Known parameters for generating synthetic data (S0, T2)
TRUE_PARS = np.array([500.0, 90.0])


# ---- ExpModel.model / model_jac ----


class TestExpModelPredictions:
    def test_model_formula(self):
        t = np.array([0.0, 40.0, 80.0, 160.0])
        model = ExpModel(t, np.ones_like(t))
        y = model.model(TRUE_PARS)
        expected = TRUE_PARS[0] * np.exp(-t / TRUE_PARS[1])
        np.testing.assert_allclose(y, expected, rtol=1e-12)

    def test_model_default_t_uses_sample_times(self):
        t = np.array([10.0, 20.0, 30.0])
        model = ExpModel(t, np.ones_like(t))
        np.testing.assert_allclose(model.model(TRUE_PARS), model.model(TRUE_PARS, t=t))

    def test_model_custom_t(self):
        t = np.array([10.0, 20.0, 30.0])
        model = ExpModel(t, np.ones_like(t))
        t2 = np.array([1.0, 2.0])
        y = model.model(TRUE_PARS, t=t2)
        assert y.shape == (2,)


class TestExpModelDerivatives:
    @pytest.fixture
    def model(self):
        t = np.linspace(0.44, 160, 12)
        y = TRUE_PARS[0] * np.exp(-t / TRUE_PARS[1])
        return ExpModel(t, y)

    def test_model_jac_vs_numerical(self, model):
        x = TRUE_PARS * 1.1
        jac = model.model_jac(x)
        eps = 1e-6
        for p in range(2):
            x_plus, x_minus = x.copy(), x.copy()
            x_plus[p] += eps
            x_minus[p] -= eps
            num_jac = (model.model(x_plus) - model.model(x_minus)) / (2 * eps)
            np.testing.assert_allclose(jac[p], num_jac, rtol=1e-4, atol=1e-8)

    def test_cost_jac_vs_numerical(self, model):
        x = TRUE_PARS * 1.1
        jac = model.cost_jac(x)
        eps = 1e-6
        num_jac = np.zeros(2)
        for p in range(2):
            x_plus, x_minus = x.copy(), x.copy()
            x_plus[p] += eps
            x_minus[p] -= eps
            num_jac[p] = (model.cost(x_plus) - model.cost(x_minus)) / (2 * eps)
        np.testing.assert_allclose(jac, num_jac, rtol=1e-4, atol=1e-8)

    def test_cost_hess_vs_numerical(self, model):
        x = TRUE_PARS * 1.1
        hess = model.cost_hess(x)
        eps = 1e-4
        num_hess = np.zeros((2, 2))
        for p in range(2):
            x_plus, x_minus = x.copy(), x.copy()
            x_plus[p] += eps
            x_minus[p] -= eps
            num_hess[p, :] = (model.cost_jac(x_plus) - model.cost_jac(x_minus)) / (
                2 * eps
            )
        np.testing.assert_allclose(hess, num_hess, rtol=1e-3, atol=1e-6)

    def test_cost_at_true_params_is_zero(self, model):
        assert model.cost(TRUE_PARS) == pytest.approx(0.0, abs=1e-20)

    def test_gradient_at_true_params_is_zero(self, model):
        np.testing.assert_allclose(model.cost_jac(TRUE_PARS), 0, atol=1e-8)


class TestExpModelFit:
    def test_recovers_parameters_from_clean_data(self):
        t = np.linspace(0.44, 160, 12)
        y = TRUE_PARS[0] * np.exp(-t / TRUE_PARS[1])
        model = ExpModel(t, y)
        hat, se = model.fit()
        np.testing.assert_allclose(hat, TRUE_PARS, rtol=1e-4)
        assert np.all(se >= 0)

    def test_fit_returns_covariance(self):
        t = np.linspace(0.44, 160, 12)
        y = TRUE_PARS[0] * np.exp(-t / TRUE_PARS[1])
        model = ExpModel(t, y)
        _, _, cov = model.fit(return_cov=True)
        assert cov.shape == (2, 2)
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_fit_with_noisy_data(self):
        rng = np.random.default_rng(42)
        t = np.linspace(0.44, 160, 12)
        y_clean = TRUE_PARS[0] * np.exp(-t / TRUE_PARS[1])
        y = y_clean + rng.normal(0, y_clean.mean() * 0.01, size=y_clean.shape)
        model = ExpModel(t, y)
        hat, _ = model.fit()
        np.testing.assert_allclose(hat, TRUE_PARS, rtol=0.05)

    def test_fit_default_init(self):
        t = np.linspace(0.44, 160, 12)
        y = TRUE_PARS[0] * np.exp(-t / TRUE_PARS[1])
        model = ExpModel(t, y)
        hat_default, _ = model.fit()
        hat_custom, _ = model.fit(init=[400.0, 100.0])
        np.testing.assert_allclose(hat_default, hat_custom, rtol=1e-4)


# ---- r2_to_oxy ----


class TestR2ToOxy:
    def test_known_conversion(self):
        # Directly reimplement Table 1 of Lu et al., 2012 for tau=10, hct=0.4
        coefs = [-13.5, 80.2, -75.9, 0.5, 3.4, 247.4]
        hct = 0.4
        r2 = 0.0125
        a = coefs[0] + coefs[1] * hct + coefs[2] * hct**2
        a_p = a - r2 * 1e3
        b = coefs[3] * hct + coefs[4] * hct**2
        c = coefs[5] * hct * (1 - hct)
        roots = np.roots([c, b, a_p])
        expected = (1 - np.max(roots)) * 100
        assert r2_to_oxy(r2, hct, tau=10) == pytest.approx(expected, rel=1e-10)

    def test_monotonic_decreasing_with_r2(self):
        hct = 0.4
        r2_vals = [0.008, 0.010, 0.0125, 0.015, 0.02]
        oxy_vals = [r2_to_oxy(r2, hct, tau=10) for r2 in r2_vals]
        assert all(
            oxy_vals[i] > oxy_vals[i + 1] for i in range(len(oxy_vals) - 1)
        )

    def test_default_tau(self):
        assert r2_to_oxy(0.0125, 0.4) == r2_to_oxy(0.0125, 0.4, tau=10)

    def test_all_valid_tau_values_run(self):
        for tau in [5, 10, 15, 20]:
            oxy = r2_to_oxy(0.0125, 0.4, tau=tau)
            assert np.isreal(oxy)

    def test_invalid_tau_raises(self):
        with pytest.raises(KeyError):
            r2_to_oxy(0.0125, 0.4, tau=7)


# ---- make_roi ----


class TestMakeRoi:
    def test_auto_selects_peak_cluster(self):
        diff_img = np.zeros((9, 9))
        diff_img[4, 4] = 5
        diff_img[4, 5] = 4
        diff_img[5, 4] = 3
        diff_img[5, 5] = 2
        roi = make_roi("auto", diff_img, n_vox=4)
        assert roi.shape == (9, 9, 1)
        assert roi.sum() == 4
        selected = set(map(tuple, np.argwhere(roi.squeeze())))
        assert selected == {(4, 4), (4, 5), (5, 4), (5, 5)}

    def test_auto_respects_n_vox(self):
        diff_img = np.zeros((9, 9))
        diff_img[4, 4] = 5
        diff_img[4, 5] = 4
        diff_img[5, 4] = 3
        diff_img[5, 5] = 2
        roi = make_roi("auto", diff_img, n_vox=2)
        assert roi.sum() == 2
        selected = set(map(tuple, np.argwhere(roi.squeeze())))
        assert selected == {(4, 4), (4, 5)}

    def test_auto_ignores_background(self):
        diff_img = np.zeros((9, 9))
        diff_img[1, 1] = 10
        roi = make_roi("auto", diff_img, n_vox=1)
        assert tuple(np.argwhere(roi.squeeze())[0]) == (1, 1)


# ---- Integration: main() end to end ----


def _make_trust_data(roi_vox, s0, t2, eff_te, eff_rep, shape_xy=(9, 9), noise_frac=0.0, seed=0):
    """Build a synthetic 4D TRUST NIfTI array where the given ROI voxels follow a
    known single-exponential decay (control - label) and everywhere else is flat."""
    eff_te_full = np.tile(eff_te, eff_rep)
    n_pairs = len(eff_te_full)
    nx, ny = shape_xy
    label_base = 1000.0
    data = np.full((nx, ny, 1, n_pairs * 2), label_base)
    rng = np.random.default_rng(seed)
    for k in range(n_pairs):
        diff_val = s0 * np.exp(-eff_te_full[k] / t2)
        for x, y, z in roi_vox:
            noise = rng.normal(0, diff_val * noise_frac) if noise_frac else 0.0
            data[x, y, z, 2 * k] = label_base
            data[x, y, z, 2 * k + 1] = label_base + diff_val + noise
    return data


ROI_VOX = [(4, 4, 0), (4, 5, 0), (5, 4, 0), (5, 5, 0)]
EFF_TE = np.array([0.44, 40, 80, 160])
EFF_REP = 3


class TestMain:
    @pytest.fixture
    def trust_files(self, tmp_path):
        data = _make_trust_data(ROI_VOX, *TRUE_PARS, EFF_TE, EFF_REP)
        affine = np.eye(4)
        trust_path = str(tmp_path / "trust.nii.gz")
        nib.Nifti1Image(data, affine).to_filename(trust_path)

        mask = np.zeros((9, 9, 1))
        for x, y, z in ROI_VOX:
            mask[x, y, z] = 1
        mask_path = str(tmp_path / "mask.nii.gz")
        nib.Nifti1Image(mask, affine).to_filename(mask_path)

        return {"trust": trust_path, "mask": mask_path, "out": str(tmp_path / "out")}

    def test_recovers_true_parameters_auto_roi(self, trust_files):
        main([trust_files["trust"], trust_files["out"], "-t1_blood", "1e12"])
        with open(f"{trust_files['out']}_pars.json", encoding="utf-8") as f:
            pars = json.load(f)
        assert pars["fitted"]["S0"]["value"] == pytest.approx(TRUE_PARS[0], rel=1e-3)
        assert pars["fitted"]["T2"]["value"] == pytest.approx(TRUE_PARS[1], rel=1e-3)

    def test_writes_expected_output_files(self, trust_files):
        main([trust_files["trust"], trust_files["out"], "-t1_blood", "1e12"])
        assert os.path.exists(f"{trust_files['out']}_pars.json")
        assert os.path.exists(f"{trust_files['out']}_fit.jpeg")
        assert os.path.exists(f"{trust_files['out']}_roi.jpeg")

    def test_explicit_mask_matches_auto_roi(self, trust_files):
        out_auto = trust_files["out"] + "_auto"
        out_mask = trust_files["out"] + "_mask"
        main([trust_files["trust"], out_auto, "-t1_blood", "1e12"])
        main(
            [
                trust_files["trust"],
                out_mask,
                "-t1_blood",
                "1e12",
                "-mask",
                trust_files["mask"],
            ]
        )
        with open(f"{out_auto}_pars.json", encoding="utf-8") as f:
            pars_auto = json.load(f)
        with open(f"{out_mask}_pars.json", encoding="utf-8") as f:
            pars_mask = json.load(f)
        assert pars_auto["fitted"]["T2"]["value"] == pytest.approx(
            pars_mask["fitted"]["T2"]["value"]
        )

    def test_flip_label_recovers_same_fit(self, tmp_path):
        # Build data with control/label swapped relative to _make_trust_data's
        # default (label, control) ordering.
        eff_te_full = np.tile(EFF_TE, EFF_REP)
        data = np.full((9, 9, 1, len(eff_te_full) * 2), 1000.0)
        for k, te in enumerate(eff_te_full):
            diff_val = TRUE_PARS[0] * np.exp(-te / TRUE_PARS[1])
            for x, y, z in ROI_VOX:
                data[x, y, z, 2 * k] = 1000.0 + diff_val
                data[x, y, z, 2 * k + 1] = 1000.0
        trust_path = str(tmp_path / "trust_flip.nii.gz")
        nib.Nifti1Image(data, np.eye(4)).to_filename(trust_path)
        mask = np.zeros((9, 9, 1))
        for x, y, z in ROI_VOX:
            mask[x, y, z] = 1
        mask_path = str(tmp_path / "mask.nii.gz")
        nib.Nifti1Image(mask, np.eye(4)).to_filename(mask_path)

        out = str(tmp_path / "out_flip")
        main(
            [
                trust_path,
                out,
                "-t1_blood",
                "1e12",
                "-mask",
                mask_path,
                "-flip_label",
            ]
        )
        with open(f"{out}_pars.json", encoding="utf-8") as f:
            pars = json.load(f)
        assert pars["fitted"]["T2"]["value"] == pytest.approx(TRUE_PARS[1], rel=1e-3)

    def test_seed_reproducibility(self, tmp_path):
        data = _make_trust_data(
            ROI_VOX, *TRUE_PARS, EFF_TE, EFF_REP, noise_frac=0.01, seed=7
        )
        trust_path = str(tmp_path / "trust.nii.gz")
        nib.Nifti1Image(data, np.eye(4)).to_filename(trust_path)
        mask = np.zeros((9, 9, 1))
        for x, y, z in ROI_VOX:
            mask[x, y, z] = 1
        mask_path = str(tmp_path / "mask.nii.gz")
        nib.Nifti1Image(mask, np.eye(4)).to_filename(mask_path)

        out1 = str(tmp_path / "out1")
        out2 = str(tmp_path / "out2")
        out3 = str(tmp_path / "out3")
        common = [
            "-t1_blood",
            "1e12",
            "-mask",
            mask_path,
            "-n_sim",
            "2000",
        ]
        main([trust_path, out1, *common, "-seed", "42"])
        main([trust_path, out2, *common, "-seed", "42"])
        main([trust_path, out3, *common, "-seed", "43"])
        with open(f"{out1}_pars.json", encoding="utf-8") as f:
            p1 = json.load(f)
        with open(f"{out2}_pars.json", encoding="utf-8") as f:
            p2 = json.load(f)
        with open(f"{out3}_pars.json", encoding="utf-8") as f:
            p3 = json.load(f)
        assert p1["fitted"]["Y"]["se"] == p2["fitted"]["Y"]["se"]
        assert p1["fitted"]["Y"]["se"] != p3["fitted"]["Y"]["se"]

    def test_hct_changes_oef(self, trust_files):
        out_low = trust_files["out"] + "_lowhct"
        out_high = trust_files["out"] + "_highhct"
        main([trust_files["trust"], out_low, "-t1_blood", "1e12", "-hct", "0.3"])
        main([trust_files["trust"], out_high, "-t1_blood", "1e12", "-hct", "0.5"])
        with open(f"{out_low}_pars.json", encoding="utf-8") as f:
            p_low = json.load(f)
        with open(f"{out_high}_pars.json", encoding="utf-8") as f:
            p_high = json.load(f)
        assert p_low["fitted"]["OEF"]["value"] != p_high["fitted"]["OEF"]["value"]
