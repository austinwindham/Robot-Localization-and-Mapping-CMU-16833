'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """

        self._alpha1 = 0.00001*10
        self._alpha2 = 0.00001*10
        self._alpha3 = 0.0001*10
        self._alpha4 = 0.0001*10

    def sample(self, var):
        return np.random.normal(0, np.sqrt(var))

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """
        x0, y0, theta0 = u_t0
        x1, y1, theta1 = u_t1

        d_rot1 = math.atan2(y1-y0, x1-x0)-theta0
        d_trans = np.sqrt((x1-x0)**2+(y1-y0)**2)
        d_rot2 = theta1-theta0-d_rot1


        d_rot1 -= self.sample( self._alpha1*d_rot1**2 +self._alpha2*d_trans**2)
        d_trans -= self.sample(self._alpha3*d_trans**2 +self._alpha4*d_rot1**2+self._alpha4*d_rot2**2)
        d_rot2 -= self.sample(self._alpha1*d_rot2**2 +self._alpha2*d_trans**2)

        theta = x_t0[-1]
        x_t0[0] += d_trans*np.cos(theta + d_rot1)
        x_t0[1] += d_trans*np.sin(theta + d_rot1)
        x_t0[-1] += (d_rot1 + d_rot2)

        # get x, y, theta
        x_t1 = np.zeros(x_t0.shape)
        x_t1[0] = x_t0[0]
        x_t1[1] = x_t0[1]
        x_t1[2] = x_t0[2]

        return x_t1

