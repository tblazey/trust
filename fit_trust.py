"""
Script to compute venous oxygenation using TRUST using method from Lu et al., 2008
"""

# Load libs
import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from numpy.polynomial.polynomial import Polynomial


class ExpModel:
    """
    Class for single exponential model
    """

    def __init__(self, t, y):
        """
        Single exponential model

        Parameters
        ----------
        t : ndarray
            Array of sample times
        y : ndarray
            Array of measured data
        """

        self.t = t
        self.y = y

    def model(self, x, t=None):
        """
        Model predictions

        .. math:: x[0] * exp(-t / x[1])

        Parameters
        ----------
        x : ndarray
            Parameter vector with parameters as above
        t : ndarray
            Times to return predictions for. If not specified then returns predictions
            for sampling times

        Returns
        -------
        y : ndarray
            Model predictions
        """

        if t is None:
            t = self.t
        return x[0] * np.exp(-t / x[1])

    def model_jac(self, x, t=None):
        """
        Gradient array for model function

        Parameters
        ----------
        x : ndarray
            Parameter vector with parameters as above
        t : ndarray
            Times to return predictions for. If not specified then returns predictions
            for sampling times

        Returns
        -------
        jac : ndarray
            Jacobian array
        """

        if t is None:
            t = self.t
        d_1 = np.exp(-t / x[1])
        d_2 = x[0] * t / x[1] ** 2 * np.exp(-t / x[1])
        return np.vstack((d_1, d_2))

    def cost(self, x):
        r"""
        Sum of squares error for model

        .. math \sum ((y - x[0] * exp(-t / x[1])) ** 2)

        Parameters
        ----------
        x : ndarray
            Parameter vector with parameters as above

        Returns
        -------
        sse : float
            Sum of squares error
        """

        return np.sum(np.power(self.y - self.model(x), 2)) * 0.5

    def cost_jac(self, x):
        """
        Jacobian vector of model cost function

        Parameters
        ----------
        x : ndarray
            Parameter vector with first parameter

        Returns
        -------
        jac : ndarray
            Jacobian of cost function
        """

        resid = self.y - self.model(x)
        d_1 = np.sum(-resid * np.exp(-self.t / x[1]))
        d_2 = np.sum(-resid * x[0] * self.t * np.exp(-self.t / x[1]) / x[1] ** 2)
        return np.array([d_1, d_2])

    def cost_hess(self, x):
        """
        Hessian matrix of model cost function

        Parameters
        ----------
        x : ndarray
            Parameter vector with first parameter

        Returns
        -------
        jac : ndarray
            Hessian of cost function
        """

        d_11 = np.sum(np.exp(-2 * self.t / x[1]))
        d_12 = np.sum(
            self.t
            * (2 * x[0] * np.exp(-self.t / x[1]) - self.y)
            * np.exp(-self.t / x[1])
            / x[1] ** 2
        )
        d_21 = d_12
        d_22 = np.sum(
            x[0]
            * self.t
            * (
                x[0] * self.t * np.exp(-self.t / x[1]) / x[1]
                - 2 * x[0] * np.exp(-self.t / x[1])
                + self.t * (x[0] * np.exp(-self.t / x[1]) - self.y) / x[1]
                + 2 * self.y
            )
            * np.exp(-self.t / x[1])
            / x[1] ** 3
        )
        return np.array([[d_11, d_12], [d_21, d_22]])

    def fit(self, return_cov=False):
        """
        Fits single exponential model to fit

        Parameters
        ----------
        return_cov : bool
            Returns covariance matrix if true

        Returns
        -------
        x_hat : ndarray
            Estimated model parameters
        x_se : ndarray
            Standard errors / standard deviation of model parameters
        x_cov : ndarray
            Covariance matrix of model parameters
        """

        # Run model fitting
        fit = minimize(
            self.cost,
            [self.y.max(), 60],
            jac=self.cost_jac,
            hess=self.cost_hess,
            method="Newton-CG",
        )

        # Get standard errors
        hess = self.cost_hess(fit.x)
        sigma_sq = self.cost(fit.x) / (self.t.shape[0] - 2)
        x_cov = sigma_sq * np.linalg.inv(hess)
        x_se = np.sqrt(np.diag(x_cov))

        if return_cov is True:
            return fit.x, x_se, x_cov
        return fit.x, x_se


def r2_to_oxy(r2, hct, tau=10):
    """
    Uses model of Lu et al., 2012 MRM to compute brain blood oxygenation given
    R2 and hematocrit

    Parameters
    ----------
    r2 : float
        Blood T2 relaxation rate in millisecond (inverse of T2)
    hct : float
        Hematocrit
    tau : float
        Inter-echo spacing of TRUST CPMG sequence. Takes 5, 10, 15, 20

    Returns
    -------
    oxy : float
        Blood oxygenation
    """

    # Parameters from Table 1  of Lu et al.
    model_coefs = {
        5: [-4.4, 39.1, -33.5, 1.5, 4.7, 167.8],
        10: [-13.5, 80.2, -75.9, -0.5, 3.4, 247.4],
        15: [-12.0, 77.7, -75.5, -6.6, 31.4, 249.4],
        20: [7.0, -9.2, 23.2, -4.5, 5.3, 310.8],
    }
    coefs = model_coefs[tau]

    # Make polynomial to convert R2 to blood oxygenation. See equations 1 - 4 of Lu et al.
    a = coefs[0] + coefs[1] * hct + coefs[2] * hct**2
    a_p = a - r2 * 1e3
    b = coefs[3] * hct + coefs[4] * hct**2
    c = coefs[5] * hct * (1 - hct)
    oxy_poly = Polynomial([a_p, b, c])
    oxy = (1 - np.max(oxy_poly.roots())) * 100

    return oxy


def make_roi(method, diff_img, n_vox=4, under_img=None):
    """
    Constructs an ROI for TRUST processsing

    Parameters
    ----------
    method : str
        Method for ROI creation. Either auto or manual.
        If manual, users selects via a plot.
    diff_img : ndarray
        Difference between labels and controls
    n_vox : int
        Number of peak voxels to use within ROI
    under_img: ndarray
        Image to show underneath difference

    Returns
    -------
    roi_dat : ndarray
        Array containing 1s within ROI
    """

    if method == "manual":

        def rect_callback(e_click, e_release):
            """
            Callback for rectangular selector
            """

        def button_callback(event):
            """
            Callback for button
            """

            if np.allclose(rect.corners[0], np.zeros(4)):
                print("Please click and drag to select an roi")
            else:
                plt.close("all")

        # Define colormap for difference image
        diff_map = plt.get_cmap("plasma")
        diff_map.set_under(alpha=0)
        diff_max = np.max(diff_img)
        diff_min = diff_max * 0.05

        # Create figure showing difference and underlying anatomy
        fig, ax = plt.subplots()
        if under_img is not None:
            ax.matshow(under_img.T, cmap="gray", origin="lower")
        ax.matshow(
            diff_img.T, cmap=diff_map, vmin=diff_min, vmax=diff_max, origin="lower"
        )

        # Create widgets
        rect = RectangleSelector(
            ax, rect_callback, interactive=True, props={"alpha": 0.5}
        )
        b_ax = fig.add_axes([0.7, 0.01, 0.1, 0.075])
        b_sub = Button(b_ax, "Submit")
        b_sub.on_clicked(button_callback)

        # Run plot and get indicies for ROI region
        plt.show()
        corners = np.array(rect.corners)
        c_min = np.floor(np.min(corners, axis=1)).astype(int)
        c_max = np.ceil(np.max(corners, axis=1)).astype(int) + 1

    else:
        # Smooth with a square kernel
        kernel = np.ones((3, 3))
        diff_conv = ndimage.convolve(diff_img, kernel)

        # Find voxel with largest signal within kernel
        idx = np.unravel_index(np.argmax(diff_conv), diff_img.shape)

        # Get indicies for ROI region
        c_min = np.maximum(np.array(idx) - 4, 0)
        c_max = np.minimum(np.array(idx) + 4, np.array(diff_img.shape) + 1)

    # Find voxels within ROI region with largest signal
    diff_sub = diff_img[c_min[0] : c_max[0], c_min[1] : c_max[1]]
    diff_sort = np.argpartition(diff_sub.flatten(), -n_vox)[-n_vox:]
    diff_idx = np.array(np.unravel_index(diff_sort, diff_sub.shape))
    diff_idx += c_min[:, np.newaxis]

    # Make roi mask
    roi_dat = np.zeros(diff_img.shape)
    roi_dat[diff_idx[0, :], diff_idx[1, :]] = 1

    return roi_dat[..., None]


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
        help="Number of repetions of effective echo times in -eff_te.",
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

    # Fit exponential model to data to get T2
    model = ExpModel(eff_te, roi_diff)
    hat, se, cov = model.fit(return_cov=True)

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
        "T2": {"value": np.round(hat[1], 5), "se": np.round(se[1], 5), "unit": "ms"},
        "S0": {"value": np.round(hat[0], 5), "se": np.round(se[0], 5), "unit": "A.U."},
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
