"""Core model and ROI logic for trust: computes venous blood oxygenation from TRUST data."""

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
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

    def fit(self, init=None, return_cov=False):
        """
        Fits single exponential model to data

        Parameters
        ----------
        return_cov : bool
            Returns covariance matrix if true
        init : ndarray
            Initial values fo fitting. If not specified it is data max and 60.

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
            init = [self.y.max(), 60]

        # Run model fitting
        fit = minimize(
            self.cost,
            init,
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
        10: [-13.5, 80.2, -75.9, 0.5, 3.4, 247.4],
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
