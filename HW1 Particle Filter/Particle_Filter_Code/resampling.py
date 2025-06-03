'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled =  np.zeros_like(X_bar)
        return X_bar_resampled

    def w_norm(self, X_bar):
        '''
        Normalize particles' weight
        '''
        wt_sum = np.sum(X_bar[:, -1])
        M = X_bar.shape[0]

        if np.abs(wt_sum) < 1e-32:
            X_bar[:, -1] = 1/X_bar.shape[0]
        else:
            X_bar[:, -1] = X_bar[:, -1] /wt_sum
            
        return X_bar

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar = self.w_norm(X_bar)
 
        X_bar_resampled = np.zeros_like(X_bar, np.float64)
     
        # initializee variables
        M = X_bar.shape[0]
        r = np.random.uniform(0, 1/M)
        i = 0
        c = X_bar[i,-1]

        # loop over particles
        for m in range(1, M+1):
            U = r+(m-1)/M
            while U >c:
                i += 1
                c += X_bar[i,-1]

            X_bar_resampled[m-1] = X_bar[i]

        return X_bar_resampled
