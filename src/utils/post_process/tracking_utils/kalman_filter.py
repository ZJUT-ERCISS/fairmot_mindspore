# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
kalman_filter
"""
# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg

# """
# Table for the 0.95 quantile of the chi-square distribution with N degrees of
# freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
# function and used as Mahalanobis gating threshold.
# """
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter:
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._update_matrix = np.eye(ndim, 2 * ndim)
        self._motion_matrix = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_matrix[i, ndim + i] = dt

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_vel = 1. / 160
        self._std_weight_pos = 1. / 20

    def initiate(self, measurement):
        """
        Create track from unassociated measurement.

        Args:
            measurement (numpy.ndarray): Bounding box coordinates (x, y, a, h) with
                center position (x, y), aspect ratio a, and height h.

        Returns:
            (numpy.ndarray, numpy.ndarray), returns the mean vector (8 dimensional) and covariance matrix
            (8x8 dimensional) of the new track. Unobserved velocities are initialized to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_pos * measurement[3],
            2 * self._std_weight_pos * measurement[3],
            1e-2,
            2 * self._std_weight_pos * measurement[3],
            10 * self._std_weight_vel * measurement[3],
            10 * self._std_weight_vel * measurement[3],
            1e-5,
            10 * self._std_weight_vel * measurement[3]]
        cov = np.diag(np.square(std))
        return mean, cov

    def predict(self, mean, cov):
        """Run Kalman filter prediction step.

        Args:
            mean (numpy.ndarray): The 8 dimensional mean vector of the object state at the previous
                time step.
            cov (numpy.ndarray): The 8x8 dimensional covariance matrix of the object state at the
                previous time step.

        Returns:
            (numpy.ndarray, numpy.ndarray), returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_vel = [
            self._std_weight_vel * mean[3],
            self._std_weight_vel * mean[3],
            1e-5,
            self._std_weight_vel * mean[3]]
        std_pos = [
            self._std_weight_pos * mean[3],
            self._std_weight_pos * mean[3],
            1e-2,
            self._std_weight_pos * mean[3]]

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # mean = np.dot(self._motion_matrix, mean)
        mean = np.dot(mean, self._motion_matrix.T)
        cov = np.linalg.multi_dot((
            self._motion_matrix, cov, self._motion_matrix.T)) + motion_cov

        return mean, cov

    def project(self, mean, cov):
        """Project state distribution to measurement space.

        Args:
            mean (numpy.ndarray): The state's mean vector (8 dimensional array).
            cov (numpy.ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            (numpy.ndarray, numpy.ndarray), returns the projected mean and covariance matrix of
            the given state estimate.

        """
        std = [
            self._std_weight_pos * mean[3],
            self._std_weight_pos * mean[3],
            1e-1,
            self._std_weight_pos * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_matrix, mean)
        cov = np.linalg.multi_dot((
            self._update_matrix, cov, self._update_matrix.T))
        return mean, cov + innovation_cov

    def multi_predict(self, mean, cov):
        """Run Kalman filter prediction step (Vectorized version).

        Args:
            mean (numpy.ndarray): The Nx8 dimensional mean matrix of the object states
                at the previous time step.
            cov (numpy.ndarray): The Nx8x8 dimensional covariance matrics of the
                object states at the previous time step.

        Returns:
            (numpy.ndarray, numpy.ndarray), returns the mean vector and covariance matrix
            of the predicted state. Unobserved velocities are initialized to 0 mean.

        """
        std_vel = [
            self._std_weight_vel * mean[:, 3],
            self._std_weight_vel * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_vel * mean[:, 3]]
        std_pos = [
            self._std_weight_pos * mean[:, 3],
            self._std_weight_pos * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_pos * mean[:, 3]]

        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_matrix.T)
        left = np.dot(self._motion_matrix, cov).transpose((1, 0, 2))
        cov = np.dot(left, self._motion_matrix.T) + motion_cov

        return mean, cov

    def update(self, mean, cov, measurement):
        """Run Kalman filter correction step.

        Args:
            mean (numpy.ndarray): The predicted state's mean vector (8 dimensional).
            cov (numpy.ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (numpy.ndarray): The 4 dimensional measurement vector (x, y, a, h),
                where (x, y) is the center position, a the aspect ratio, and h the height
                of the bounding box.

        Returns:
            (numpy.ndarray, numpy.ndarray), returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, cov)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(cov, self._update_matrix.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_cov = cov - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_cov

    def gating_distance(self, mean, cov, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Args:
            mean (numpy.ndarray): Mean vector over the state distribution (8 dimensional).
            cov (numpy.ndarray): Covariance of the state distribution (8x8 dimensional).
            measurements (numpy.ndarray): An Nx4 dimensional matrix of N measurements,
                each in format (x, y, a, h) where (x, y) is the bounding box center position,
                a the aspect ratio, and h the height.
            only_position (Optional[bool]): If True, distance computation is done with respect to
                the bounding box center position only.

        Returns:
            numpy.ndarray, returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and `measurements[i]`.

        """
        mean, cov = self.project(mean, cov)
        if only_position:
            mean, cov = mean[:2], cov[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            res = np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(cov)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            res = np.sum(z * z, axis=0)
        else:
            raise ValueError('invalid distance metric')
        return res
