"""Command-line interface for trust: computes venous blood oxygenation from TRUST data."""

import argparse
import json

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from trust.core import ExpModel, make_roi, r2_to_oxy


def create_parser():
    """
    Creates argparse argument parser for TRUST processing script
    """

    parser = argparse.ArgumentParser(
        description="Computes venous blood oxygenation (Y) from TRUST data.",
        epilog="By default temporal dimension of TRUST image should be as follows: "
        "([label, control].1 ... [label, control].N_eff_te) * eff_rep",
    )
    parser.add_argument("trust", type=str, help="Nifti image containing TRUST.")
    parser.add_argument("out", type=str, help="Root for file outputs.")
    parser.add_argument(
        "-mask",
        type=str,
        help="Nifti image with roi mask. If not specified will find region with "
        "highest signal (see -roi_method).",
    )
    parser.add_argument(
        "-eff_te",
        type=float,
        default=[0.44, 40, 80, 160],
        nargs="+",
        help="Effective echo times (ms). Defaults are 0.44, 40, 80, 160.",
    )
    parser.add_argument(
        "-eff_rep",
        type=int,
        default=3,
        help="Number of repetions of effective echo times in -eff_te. Default is 3.",
    )
    parser.add_argument(
        "-tau",
        choices=[5, 10, 15, 20],
        default=10,
        type=int,
        help="Inter-echo spacing of TRUST CPMG sequence (ms). Default is 10.",
    )
    parser.add_argument(
        "-inv_time",
        type=float,
        default=1020,
        help="Inversion time (ms). Default is 1020.",
    )
    parser.add_argument(
        "-t1_blood",
        type=float,
        default=1624,
        help="T1 relaxation time for blood (ms). Default is 1624.",
    )
    parser.add_argument(
        "-flip_label",
        action="store_true",
        help="Changes expected image order from Label, Control to Control, Label.",
    )
    parser.add_argument(
        "-hct", type=float, default=0.4, help="Hematocrit fraction. Default is 0.4"
    )
    parser.add_argument(
        "-n_sim",
        type=int,
        default=10000,
        help="Number of simulations for computing standard error of blood "
        "oxygenation (Y) and OEF. Default is 10,000.",
    )
    parser.add_argument(
        "-art_oxy",
        type=float,
        default=100,
        help="Arterial oxygen content (%%) for OEF. Default is 100.",
    )
    parser.add_argument(
        "-roi_method",
        type=str,
        default="auto",
        choices=["manual", "auto"],
        help="Method for ROI generation if -mask is not specified. Takes manual or auto. "
        "If manual users selects a square region from a plot. If auto than program "
        "finds region whose 9 nearest neighbors have the greatest average signal. "
        "In either case, the -n_vox largest voxels within the ROI are used. "
        "Default is auto.",
    )
    parser.add_argument(
        "-n_vox",
        type=int,
        default=4,
        help="Number of voxels to select within region when using -roi-method. "
        "Default is 4.",
    )
    parser.add_argument("-seed", type=int, help="Seed for standard error simulations.")

    return parser


def main(argv=None):
    """
    Run processing script
    """

    # Run parser
    parser = create_parser()
    args = parser.parse_args(argv)

    # Get array of effective echo timse
    eff_te = np.tile(args.eff_te, args.eff_rep)
    min_eff_idx = np.argwhere(eff_te == np.min(eff_te)).flatten()

    # Load in image data
    trust_hdr = nib.load(args.trust)
    trust_dat = trust_hdr.get_fdata()

    # Compute label control difference
    trust_diff = trust_dat[..., 1::2] - trust_dat[..., 0::2]
    if args.flip_label is True:
        trust_diff *= -1

    # Either load in a mask or create one
    if args.mask is not None:
        roi_hdr = nib.load(args.mask)
        roi_dat = roi_hdr.get_fdata()

    else:
        # Make average of minimum echo time images
        min_eff_diff = np.mean(trust_diff[..., min_eff_idx], axis=-1).squeeze()
        min_eff_img = np.mean(trust_dat[..., min_eff_idx], axis=-1).squeeze()

        # Make image for mask
        roi_dat = make_roi(
            args.roi_method, min_eff_diff, n_vox=args.n_vox, under_img=min_eff_img
        )

    # Apply mask to trust data
    roi_msk = np.broadcast_to(roi_dat[..., None] == 0, trust_diff.shape)
    roi_diff = np.ma.mean(np.ma.array(trust_diff, mask=roi_msk), axis=(0, 1, 2)).data

    # Adjust for influence of t1 of blood. In Lu et al., 2018 this is done after fitting
    # (equation 5). Here we remove those components for the data (equation 4)
    roi_diff *= np.exp((args.inv_time - eff_te) / args.t1_blood)

    # Use linear regression to get initial estimate
    if np.min(roi_diff) > 0:
        A = np.stack((np.ones_like(eff_te), -eff_te), axis=1)
        coef, _, _, _ = np.linalg.lstsq(A, np.log(roi_diff), rcond=None)
        init = np.array([np.exp(coef[0]), 1 / coef[1]])
    else:
        init = np.array([np.max(roi_diff), 60])

    # Fit exponential model to data to get T2
    model = ExpModel(eff_te, roi_diff)
    hat, se, cov = model.fit(return_cov=True, init=init)

    # Get standard error for R2
    r2_diff = -1e3 / hat[1] ** 2
    r2_se = np.sqrt(r2_diff * cov[1, 1] * r2_diff)

    # Convert T2/R2 to blood oxygenation
    oxy = r2_to_oxy(1 / hat[1], args.hct, tau=args.tau)
    oef = (args.art_oxy - oxy) / args.art_oxy

    # Run simulation to get standard errors
    r2 = 1e3 / hat[1]
    if args.seed is not None:
        np.random.seed(args.seed)
    r2_sim = np.random.normal(loc=r2, scale=r2_se, size=args.n_sim) * 1e-3
    oxy_sim = np.zeros_like(r2_sim)
    oef_sim = np.zeros_like(r2_sim)
    for idx, r2_s in enumerate(r2_sim):
        oxy_sim[idx] = r2_to_oxy(r2_s, args.hct, tau=args.tau)
        oef_sim[idx] = (args.art_oxy - oxy_sim[idx]) / args.art_oxy
    oxy_se = np.std(oxy_sim)
    oef_se = np.std(oef_sim)

    # Make dictionaries with fitted and input parameters
    fit_dic = {
        "T2": {
            "init": np.round(init[1], 5),
            "value": np.round(hat[1], 5),
            "se": np.round(se[1], 5),
            "unit": "ms",
        },
        "S0": {
            "init": np.round(init[0], 5),
            "value": np.round(hat[0], 5),
            "se": np.round(se[0], 5),
            "unit": "A.U.",
        },
        "R2": {"value": np.round(r2, 5), "se": np.round(r2_se, 5), "unit": "1/s"},
        "Y": {"value": np.round(oxy, 5), "se": np.round(oxy_se, 5), "unit": "%"},
        "OEF": {"value": np.round(oef, 5), "se": np.round(oef_se, 5), "unit": "frac."},
    }
    in_dic = vars(args)
    par_dic = {"input": in_dic, "fitted": fit_dic}
    with open(f"{args.out}_pars.json", "w", encoding="utf-8") as j_id:
        json.dump(par_dic, j_id, indent=4)

    # Compute model predictions and 99% confidence intervals
    n_p = 100
    te_hat = np.linspace(eff_te.min(), eff_te.max(), n_p)
    diff_hat = model.model(hat, t=te_hat)
    diff_jac = model.model_jac(hat, t=te_hat)
    diff_se = np.zeros(n_p)
    for idx in range(n_p):
        diff_se[idx] = np.sqrt(diff_jac[:, idx] @ cov @ diff_jac[:, idx])
    diff_up = diff_hat + diff_se * 2.575
    diff_dn = diff_hat - diff_se * 2.575

    # Make a QA plot
    plt.grid()
    plt.scatter(eff_te, roi_diff, c="black")
    plt.plot(te_hat, diff_hat, lw=3)
    plt.xlabel("Effective TE (ms)", fontweight="bold")
    plt.ylabel(r"$\Delta$ Signal", fontweight="bold")
    plt.fill_between(te_hat, y1=diff_dn, y2=diff_up, alpha=0.6, color="#999999")
    plt.title("Model Fit", fontweight="bold")
    plt.savefig(f"{args.out}_fit.jpeg", dpi=200, bbox_inches="tight")
    plt.close("all")

    # Make roi plot
    min_eff_mean = np.mean(trust_dat[..., 0::2][..., min_eff_idx], axis=-1).squeeze()
    fig, ax = plt.subplots()
    ax.matshow(min_eff_mean.T, cmap="gray", origin="lower")
    roi_map = plt.get_cmap("Reds")
    roi_map.set_under(alpha=0)
    ax.matshow(roi_dat.squeeze().T, cmap=roi_map, vmin=0.5, vmax=1, origin="lower")
    ax.set_title("TRUST ROI", fontweight="bold")
    ax.axis("off")
    fig.savefig(f"{args.out}_roi.jpeg", dpi=200, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    main()
