"""
ase testing
"""

# load libs
import argparse
import json
import sys
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, nnls
from tqdm import tqdm


class AseModel:
    """
    Class for ASE model fitting
    """

    def __init__(self, tau, te, y, tau_cut=15, default=None, bounds=None):
        """
        ASE model (See An and Lin, 2003)

        Parameters
        ----------
        tau : ndarray
            Time interval from TE / 2 to center of 180 degree pulse (ms)
        te : ndarray
            Echo time (ms)
        y : ndarray
            Array of measured data
        tau_cut : float
            Cutoff point for short echo time period (ms)
        default : list
            List containing default values for parameter search
        bounds : list
            A 2d list with each sublist containing min and max for each parameter
        """

        self.tau = tau
        self.te = te
        self.y = y
        self.tau_cut = tau_cut
        self.n = tau.shape[0]
        if default is None:
            self.default = [2000, 0.1, 0.07, 100]
        if bounds is None:
            self.bounds = [[0, np.inf], [0, 1], [0.01, 0.5], [10, 5000]]
        self.__create_design()
        self.names = ["s0", "lambda", "omega", "t2"]

    def model(self, x, tau=None, te=None):
        """
        ASE model predictions given coefficient vector

        Parameters
        ----------
        x : ndarray
            Parameter vector where x[0] = s_0
                                   x[1] = lmbda
                                   x[2] = delta_omega
                                   x[3] = t_2 (ms)
        tau : ndarray
            Interval times to return predictions for (ms). If not specified then
            returns predictions for sampling taus.
        te : ndarray
            Echo times to return predictions for (ms). If not specified then
            returns predictions for sampling TEs.

        Returns
        -------
        y_hat : ndarray
            Model predictions
        """

        # Use sampling times if prediction times aren't specified
        if tau is None:
            tau = self.tau
        if te is None:
            te = self.te
        if tau.shape != te.shape:
            raise ValueError("Dimensions of te and tau must match")

        # Prep for model predictions
        s_0, lmbda, omega, t_2 = x
        y_hat = np.zeros_like(tau)

        # Get model predictions from piecewise model
        for idx, [tau_i, te_i] in enumerate(zip(tau, te)):
            if tau_i <= self.tau_cut:
                y_hat[idx] = (
                    s_0
                    * np.exp(-0.3 * lmbda * (omega * tau_i) ** 2)
                    * np.exp(-te_i / t_2)
                )
            else:
                y_hat[idx] = (
                    s_0 * np.exp(-lmbda * omega * tau_i + lmbda) * np.exp(-te_i / t_2)
                )

        return y_hat

    def model_jac(self, x, tau=None, te=None):
        """
        Gradient array for model function

        Parameters
        ----------
        x : ndarray
            Parameter vector with parameters as above
        tau : ndarray
            Interval times to return predictions for (ms). If not specified then
            returns predictions for sampling taus.
        te : ndarray
            Echo times to return predictions for (ms). If not specified then
            returns predictions for sampling TEs.

        Returns
        -------
        jac : ndarray
            Jacobian array for model function
        """
        s_0, lmbda, omega, t_2 = x

        # Use sampling times if prediction times aren't specified
        if tau is None:
            tau = self.tau
        if te is None:
            te = self.te
        if tau.shape != te.shape:
            raise ValueError("Dimensions of te and tau must match")

        jac = np.zeros((4, self.n))
        for idx, [tau_i, te_i] in enumerate(zip(tau, te)):
            if tau_i <= self.tau_cut:
                tau_omega_sq = (omega * tau_i) ** 2
                c_term = np.exp(-te_i / t_2 - 0.3 * lmbda * tau_omega_sq)
                jac[0, idx] = c_term
                jac[1, idx] = -0.3 * s_0 * tau_omega_sq * c_term
                jac[2, idx] = -0.6 * s_0 * lmbda * tau_i**2 * omega * c_term
                jac[3, idx] = s_0 * te_i * c_term / t_2**2
            else:
                c_term = np.exp(-te_i / t_2 - lmbda * tau_i * omega + lmbda)
                jac[0, idx] = c_term
                jac[1, idx] = s_0 * (1 - tau_i * omega) * c_term
                jac[2, idx] = -s_0 * lmbda * tau_i * c_term
                jac[3, idx] = s_0 * te_i * c_term / t_2**2

        return jac

    def model_hess(self, x, tau=None, te=None):
        """
        Hessian array for model function

        Parameters
        ----------
        x : ndarray
            Parameter vector with parameters as above
        tau : ndarray
            Interval times to return predictions for (ms). If not specified then
            returns predictions for sampling taus.
        te : ndarray
            Echo times to return predictions for (ms). If not specified then
            returns predictions for sampling TEs.

        Returns
        -------
        hess : ndarray
            hesssian array for model function
        """
        s_0, lmbda, omega, t_2 = x

        # Use sampling times if prediction times aren't specified
        if tau is None:
            tau = self.tau
        if te is None:
            te = self.te
        if tau.shape != te.shape:
            raise ValueError("Dimensions of te and tau must match")

        # Fill in hessian terms
        hess = np.zeros((4, 4, tau.shape[0]))
        for idx, [tau_i, te_i] in enumerate(zip(tau, te)):
            tau_omega_sq = (omega * tau_i) ** 2
            if tau_i <= self.tau_cut:
                c_term = np.exp(-te_i / t_2 - 0.3 * lmbda * tau_omega_sq)
                hess[0, 1, idx] = -0.3 * tau_omega_sq * c_term
                hess[0, 2, idx] = -0.6 * lmbda * tau_i**2 * omega * c_term
                hess[0, 3, idx] = (
                    te_i * np.exp(-te_i / t_2 - 0.3 * lmbda * tau_omega_sq) / t_2**2
                )
                hess[1, 1, idx] = 0.09 * s_0 * tau_omega_sq**2 * c_term
                hess[1, 2, idx] = (
                    s_0
                    * tau_i**2
                    * omega
                    * (0.18 * lmbda * tau_omega_sq - 0.6)
                    * c_term
                )
                hess[1, 3, idx] = -0.3 * s_0 * te_i * tau_omega_sq * c_term / t_2**2
                hess[2, 2, idx] = (
                    s_0
                    * lmbda
                    * tau_i**2
                    * (0.36 * lmbda * tau_omega_sq - 0.6)
                    * c_term
                )
                hess[2, 3, idx] = (
                    -0.6 * s_0 * te_i * lmbda * tau_i**2 * omega * c_term / t_2**2
                )
            else:
                tau_omega_one = tau_i * omega - 1
                c_term = np.exp(-te_i / t_2 - lmbda * tau_omega_one)
                hess[0, 1, idx] = -tau_omega_one * c_term
                hess[0, 2, idx] = -lmbda * tau_i * c_term
                hess[0, 3, idx] = te_i * c_term / t_2**2
                hess[1, 1, idx] = s_0 * tau_omega_one**2 * c_term
                hess[1, 2, idx] = s_0 * tau_i * (lmbda * tau_omega_one - 1) * c_term
                hess[1, 3, idx] = (
                    s_0
                    * te_i
                    * (-tau_omega_one)
                    * np.exp(-te_i / t_2 - lmbda * tau_i * omega + lmbda)
                    / t_2**2
                )
                hess[2, 2, idx] = s_0 * lmbda**2 * tau_i**2 * c_term
                hess[2, 3, idx] = -s_0 * te_i * lmbda * tau_i * c_term / t_2**2
            hess[3, 3, idx] = s_0 * te_i * (te_i - 2 * t_2) * c_term / t_2**4

        # Make matrix symmetrical
        hess[1, 0, :] = hess[0, 1, :]
        hess[2, 0, :] = hess[0, 2, :]
        hess[2, 1, :] = hess[1, 2, :]
        hess[3, 0, :] = hess[0, 3, :]
        hess[3, 1, :] = hess[1, 3, :]
        hess[3, 2, :] = hess[2, 3, :]

        return hess

    def cost_jac(self, x):
        """
        Gradient array for cost function

        Parameters
        ----------
        x : ndarray
            Parameter vector with parameters as above

        Returns
        -------
        jac : ndarray
            Jacobian array for cost function
        """

        # Get jacobian array for model function
        jac_model = self.model_jac(x)

        # Compute residuals
        resid = self.y - self.model(x)

        # Add residual term and sum to get jacobian for cost function
        jac = np.sum(-jac_model * resid[np.newaxis, :], axis=1)

        return jac

    def cost_hess(self, x):
        """
        Hessian matrix for cost function

        Parameters
        ----------
        x : ndarray
            Parameter vector with parameters as above

        Returns
        -------
        hess : ndarray
            Hessian array for cost function
        """

        # Get derivatives for model functions
        jac_model = self.model_jac(x)
        hess_model = self.model_hess(x)

        # Compute residuals
        resid = self.y - self.model(x)

        # Add residual term and sum to get jacobian for cost function
        hess = np.sum(
            np.einsum("ik,jk->ijk", jac_model, jac_model) - resid * hess_model, axis=2
        )

        return hess

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

    def fit(self, init=None, return_cov=False):
        """
        Fits single exponential model to data

        Parameters
        ----------
        return_cov : bool
            Returns covariance matrix if true
        init : ndarray
            Initial values fo fitting. If not specified uses a linear estimate

        Returns
        -------
        x_hat : ndarray
            Estimated model parameters
        x_se : ndarray
            Standard errors / standard deviation of model parameters
        x_cov : ndarray
            Covariance matrix of model parameters
        """

        if init is None:
            init = self.init_par()

        # Run model fitting
        fit = minimize(
            self.cost,
            init,
            jac=self.cost_jac,
            bounds=self.bounds,
        )

        # Get standard errors
        hess = self.cost_hess(fit.x)
        sigma_sq = self.cost(fit.x) / (self.y.shape[0] - fit.x.shape[0])
        x_cov = sigma_sq * np.linalg.inv(hess)
        x_se = np.sqrt(np.diag(x_cov))

        if return_cov is True:
            return fit.x, x_se, x_cov
        return fit.x, x_se

    def __create_design(self):
        """
        Creates a design matrix for linear version of model
        """

        # Initialize design matrix
        self.A = np.zeros((self.n, 5))

        # Get masks for short and term times
        short_mask = self.tau <= self.tau_cut
        long_mask = self.tau > self.tau_cut

        # Fill in design matrix
        self.A[short_mask, 0] = 1
        self.A[long_mask, 1] = 1
        self.A[short_mask, 2] = -self.tau[short_mask] ** 2
        self.A[long_mask, 3] = -self.tau[long_mask]
        self.A[:, 4] = -self.te

    def __convert_pars(self, lin_x):
        """
        Converts linearized model coefficients to nonlinear

        Parameters
        ----------
        lin_x : ndarray
            Coefficients from fitting the linear model

        Returns
        -------
        x : ndarray
            Coefficients in nonlinear model format (s_0, lmbda, omega, t_2)
        """

        if lin_x.ndim == 1:
            x = np.zeros(lin_x.shape[0] - 1)
        else:
            x = np.zeros((lin_x.shape[0] - 1, lin_x.shape[1]))
        lin_xp = np.maximum(lin_x, 1e-10)
        x[0] = np.exp(lin_xp[0])
        x[1] = lin_xp[1] - lin_xp[0]
        x[2] = lin_x[3] / x[1]
        x[3] = 1 / lin_xp[4]

        return x

    def lin_fit(self, return_nonlin=True):
        """
        Fits a linearized version of the model to data

        Parameters
        ----------
        return_nonlin : bool
            Returns coefficients in nonlinear format (s_0, lmbda, omega, t_2) if True
            Otherwise, returns linear regression coefficients:
                log(s_0)
                log(s_0) - lmbda
                0.3 * lmbda * omega **2
                lmbda * omega
                1 / t_2
        Returns
        -------
        x : ndarray
            Coefficients in nonlinear
        """

        # Run model fitting
        if self.y.ndim > 1:
            n_vox = self.y.shape[0]
            coef = np.zeros((5, n_vox))
            for idx in tqdm(range(n_vox), desc="Initial linear fitting"):
                coef[:, idx], _ = nnls(self.A, np.log(self.y[idx, :]))
        else:
            coef, _ = nnls(self.A, np.log(self.y))

        if return_nonlin:
            return self.__convert_pars(coef)
        return coef

    def check_init(self, init):
        """
        Checks if proposed initial values are within bounds.
        If not, it returns default values for class

        Parameters
        ----------
        init : ndarray
            Proposed initial values

        Returns
        -------
        init_valid : ndarray
            Initial values within bounds
        """

        # Replace initial values that lie outside of bounds with defaults
        init_valid = np.copy(init)
        if self.bounds is not None:
            for idx in range(init.shape[0]):
                if init[idx] < self.bounds[idx][0] or init[idx] > self.bounds[idx][1]:
                    init_valid[idx] = self.default[idx]

        return init_valid

    def init_par(self):
        """
        Compute initial values for nonlinear fitting

        Returns
        -------
        init : ndarray
            Initial parameter values
        """

        # Use linear fit as estimate for nonlinear one
        return self.check_init(self.lin_fit())


def create_parser():
    """
    Creates argparse argument parser for ASE processing
    """

    parser = argparse.ArgumentParser(description="Computes OEF from ASE data.")
    parser.add_argument(
        "-ase",
        required=True,
        type=str,
        nargs="+",
        help="Nifti image(s) containing ASE data. Specify one image for each echo.",
    )
    parser.add_argument(
        "-te",
        required=True,
        type=float,
        nargs="+",
        help="Echo time for each echo (ms).",
    )
    parser.add_argument(
        "-tau", required=True, type=str, help="List containing time offsets (ms.)"
    )
    parser.add_argument(
        "-hct", default=0.4, type=float, help="Hematocrit (fraction). Default is 0.4"
    )
    parser.add_argument("-out", required=True, type=str, help="Root for file outputs.")
    parser.add_argument(
        "-mask", type=str, help="Nifti image with mask for valid voxels."
    )
    parser.add_argument(
        "-b0", default=3, type=float, help="Field strength (T). Default is 3."
    )
    parser.add_argument(
        "-chi",
        default=0.19,
        type=float,
        help="Susceptibility difference between fully oxygenated and fully "
        "deoxygengated blood (ppm per Hct). Default is 0.19",
    )
    parser.add_argument(
        "-hct_ratio",
        default=0.85,
        type=float,
        help="Ratio of large to small vessel hematocrit. Default is 0.85.",
    )
    parser.add_argument(
        "-tau_cuts",
        default=[15, 30],
        nargs=2,
        type=float,
        help="Time cutoffs (ms) for short and long tau portions of the fitting. "
        "Default is 15 and 30.",
    )
    parser.add_argument(
        "-n_skip",
        default=8,
        type=int,
        help="Number of frames to skip at the end of each image. Default is 8.",
    )
    parser.add_argument(
        "-motion_mask",
        type=str,
        help="List containing which frames we should skip due to motion. "
        "Invalid frames are specificed with a 0, valid with a 1.",
    )
    parser.add_argument(
        "-write_lin", action="store_true", help="Write out linear estimates"
    )
    parser.add_argument(
        "-wb_only", action="store_true", help="Only perform whole-brain estimate"
    )

    return parser


def delta_omega_to_oef(hat, se=None, chi=0.19, hct=0.4, hct_ratio=0.85, b0=3):
    """
    Converts output of nonlinear fitting to OEF

    Parameters
    ----------
    hat : float
        Frequency shift caused by magnetic susceptibility
    se : float
        Standard error of hat
    chi : float
        Susceptibility difference between fully oxygenated and fully
        deoxygengated blood (ppm per Hct)
    hct : float
        Hematocrit (fraction)
    hct_ratio : float
        Ratio of small to large vessel hematocrit
    b0 : float
        Field strength

    Returns
    -------
    oef : float
        Oxygen extract fraction
    oef_se : float
        Standard error for oxygen extraction fraction
    """

    # Convert oef
    conv_frac = 4 / 3 * np.pi * 267.522 * chi * hct * hct_ratio * b0
    oef_hat = hat / conv_frac * 1e3

    # Convert standard error if it is supplied
    if se is not None:
        oef_se = se / conv_frac * 1e3
        return oef_hat, oef_se
    return oef_hat


def write_masked_image(img_mskd, mask, affine, out_path):
    """
    Writes out masked data to an image

    Parameters
    ----------
    img_mskd : ndarray
        Array containing masked data
    mask : ndarray
        Array containing the mask originally applied to `img`
    affine : ndarray
        Transformation from ijk (voxels) to xyz (world)
    out_path : str
        Name to write file to. Extension is added internally.
    """

    img = np.zeros(mask.shape + img_mskd.shape[3:])
    img[mask, ...] = img_mskd
    nib.Nifti1Image(img, affine).to_filename(out_path + ".nii.gz")


def main(argv=None):
    """
    Run processing script
    """

    # Run parser
    parser = create_parser()
    args = parser.parse_args(argv)

    # Load in image headers
    hdr_list = []
    for ase_path in args.ase:
        hdr_list.append(nib.load(ase_path))
    n_img = len(hdr_list)

    # Load in tau times
    tau = np.loadtxt(args.tau)
    n_tau = tau.shape[0]

    # Input checking
    for hdr in hdr_list:
        if hdr.shape != hdr_list[0].shape:
            raise ValueError("ASE image dimensions do not match")
    if len(args.ase) != len(args.te):
        raise ValueError("Number of echo times must match the number of input images")
    if hdr_list[0].shape[3] != tau.shape[0]:
        raise ValueError("Number of tau times does not match number of image frames")

    # Mask setup
    if args.mask is not None:
        mask_hdr = nib.load(args.mask)
        if mask_hdr.shape[0:3] != hdr_list[0].shape[0:3]:
            raise ValueError("Mask does not match ase image dimensions")
        mask_data = mask_hdr.get_fdata() == 1
    else:
        mask_data = np.ones(hdr_list[0].shape[0:3], dtype=bool)

    # Setup full timing arrays
    tau_all = np.tile(tau, n_img).reshape((n_tau, n_img), order="F")
    te_all = np.repeat(args.te, n_tau)
    for idx in range(1, n_img):
        tau_all[:, idx] += args.te[idx] - args.te[0]

    # Determine which frames we are using
    frame_mask = np.logical_or(tau_all <= args.tau_cuts[0], tau_all >= args.tau_cuts[1])
    frame_mask[-args.n_skip :, :] = False
    frame_mask = frame_mask.flatten(order="F")

    # Remove motion frames if necessary
    if args.motion_mask is not None:
        motion_mask = np.loadtxt(args.motion_mask) == 1
        frame_mask = np.logical_and(frame_mask, motion_mask)

    # Extract the data we need
    img_data = np.zeros((hdr_list[0].shape + (n_img,)))
    for idx, hdr in enumerate(hdr_list):
        img_data[..., idx] = hdr.get_fdata()
    img_data = img_data.reshape((hdr_list[0].shape[0:3] + (n_img * n_tau,)), order="F")

    # Remove the frames we are not using
    tau_mskd = tau_all.flatten(order="F")[frame_mask]
    te_mskd = te_all[frame_mask]
    img_mskd = img_data[mask_data, ...][..., frame_mask]

    # Fit the whole-brain curve
    wb_avg = np.mean(img_mskd, axis=0)
    wb_model = AseModel(tau_mskd, te_mskd, wb_avg)
    wb_init = wb_model.lin_fit()
    wb_hat, wb_se, wb_cov = wb_model.fit(init=wb_init, return_cov=True)

    # Compute oef from parameter fits
    wb_oef, wb_oef_se = delta_omega_to_oef(
        wb_hat[2],
        wb_se[2],
        chi=args.chi,
        hct=args.hct,
        hct_ratio=args.hct_ratio,
        b0=args.b0,
    )

    # Make dictionaries with fitted and input parameters
    fit_dic = {
        "S0": {
            "init": np.round(wb_init[0], 5),
            "value": np.round(wb_hat[0], 5),
            "se": np.round(wb_se[0], 5),
            "unit": "A.U.",
        },
        "lambda": {
            "init": np.round(wb_init[1], 5),
            "value": np.round(wb_hat[1], 5),
            "se": np.round(wb_se[1], 5),
            "unit": "fraction",
        },
        "delta_omega": {
            "init": np.round(wb_init[2], 5),
            "value": np.round(wb_hat[2], 5),
            "se": np.round(wb_se[2], 5),
            "unit": "radians/s",
        },
        "T2": {
            "init": np.round(wb_init[3], 5),
            "value": np.round(wb_hat[3], 5),
            "se": np.round(wb_se[3], 5),
            "unit": "ms",
        },
        "OEF": {
            "value": np.round(wb_oef, 5),
            "se": np.round(wb_oef_se, 5),
            "unit": "fraction",
        },
    }
    in_dic = vars(args)
    par_dic = {"input": in_dic, "fitted": fit_dic}
    with open(f"{args.out}_pars.json", "w", encoding="utf-8") as j_id:
        json.dump(par_dic, j_id, indent=4)
        
    # Only perform whole brain fitting if requested
    if args.wb_only is True:
        sys.exit()

    # Run linear fitting
    lin_model = AseModel(tau_mskd, te_mskd, img_mskd)
    lin_hat = lin_model.lin_fit()
    lin_oef = delta_omega_to_oef(lin_hat[2])

    # Write out linear images
    if args.write_lin:
        for i in range(5):
            if i < 4:
                write_masked_image(
                    lin_hat[i, :],
                    mask_data,
                    hdr_list[0].affine,
                    f"{args.out}_{lin_model.names[i]}_lin",
                )
            else:
                write_masked_image(
                    lin_oef, mask_data, hdr_list[0].affine, f"{args.out}_oef_lin"
                )

    # Create empty parameters for results
    n_vox = img_mskd.shape[0]
    par_hat_mskd = np.zeros((5, n_vox))
    par_se_mskd = np.zeros((5, n_vox))

    # Loop through voxels
    for idx in tqdm(range(n_vox), desc="Nonlinear fitting"):
        # Fit current voxel
        vox_model = AseModel(tau_mskd, te_mskd, img_mskd[idx, :])
        par_init = vox_model.check_init(lin_hat[:, idx])
        par_hat, par_se = vox_model.fit()

        # Get oef
        oef_hat, oef_se = delta_omega_to_oef(
            wb_hat[2],
            wb_se[2],
            chi=args.chi,
            hct=args.hct,
            hct_ratio=args.hct_ratio,
            b0=args.b0,
        )

        # Save estimates
        par_hat_mskd[0:4, idx] = par_hat
        par_hat_mskd[4, idx] = oef_hat
        par_se_mskd[0:4, idx] = par_se
        par_se_mskd[4, idx] = oef_se


if __name__ == "__main__":
    main()
