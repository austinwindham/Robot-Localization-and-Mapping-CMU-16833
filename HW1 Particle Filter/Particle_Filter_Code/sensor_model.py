'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        
        self._z_hit = 0.5
        self._z_short = 0.025
        self._z_max = 0.015
        self._z_rand = 0.1

        self._sigma_hit = 100
        self._lambda_short = 0.1 


        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 2

        # parameters
        self._resolution = 10
        self._occ_size = occupancy_map.shape
        self._mask_map = occupancy_map > self._min_probability
        self._map = occupancy_map
        self._max_range = 8183

        self._table_path = "ray_cast_table.npy"
        self.check_table()

        self._ray_map = np.load("./ray_cast_table.npy")
        self._angle_step = 5

    def check_table(self):
        import os.path

        if not(os.path.isfile(self._table_path)):
            print("Stop and download the ray cast table at this link")
            print("https://drive.google.com/drive/folders/1-zmBDazQV7IW6SxhTmfTCy09GJqkTAcZ?usp=sharing")
            self.pre_compute(self.occupancy_map)


    def ray_casting(self, b_t):
        '''
        ray cast based on the beam coordinates
        '''
        x, y, th = b_t

        res = self._resolution
        i = round(x/res)
        j = round(y/res)
        
        # Extind ray until it hits obstacle
        while (0 <= i < 800) and (0 <= j < 800) and (self._mask_map[j][i] != 1):
        
            x += res*np.cos(th)
            y += res*np.sin(th)

            i = round(x/res)
            j = round(y/res)

        dist = np.sqrt((x-b_t[0])**2+(y-b_t[1])**2)

        return dist


    def get_ray(self, b_t):
        '''
        Read RayCasting direct from look up table
        '''
        x, y, th = b_t
        res = self._resolution
        m = round(x/res)
        n = round(y/res)

        degree = round(th*180/np.pi)
        if degree < -359 or degree > 359:
            degree = degree%360
        
        return self._ray_map[n, m, degree]


    # get p_hit
    def p_hit(self, zti_k, zti_k_star):

        if 0 <= zti_k <= self._max_range:
            p = np.exp(-0.5 * (zti_k - zti_k_star) ** 2 / (self._sigma_hit) ** 2)/self._sigma_hit
        else:
            p = 0.000001

        return p

    # get p_short
    def p_short(self, zti_k, zti_k_star):

        if 0 <= zti_k <= zti_k_star:
            eta = 1
            p = eta*(self._lambda_short * np.exp(-self._lambda_short * zti_k))
        else:
            p = 0.000001

        return p

    # get p_max
    def p_max(self, zti_k):

        if zti_k >= self._max_range-20:
            p = 1
        else:
            p = 0.000001

        return p
    
    # get p_rand
    def p_rand(self, zti_k):

        if 0 <= zti_k <= self._max_range:
            p = 1/self._max_range
        else:
            p = 0.000001

        return p

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        
        b_t = np.array([x_t1[0]+25*np.cos(x_t1[2]), x_t1[1]+25*np.sin(x_t1[2]), x_t1[2]])
        theta_b = b_t[-1]
        q = 0
        
        angle_step = self._angle_step
        z_t1_arr = z_t1_arr[::angle_step]

        # compare ray cast and expected
        for deg, z in enumerate(z_t1_arr):

            rad = (deg*angle_step-90+1)*np.pi/180
            b_t[-1] = theta_b+rad

            z_exp = self.get_ray(b_t)
 
            q += np.log(self._z_hit*self.p_hit(z, z_exp) + \
                        self._z_short*self.p_short(z, z_exp) + \
                        self._z_max*self.p_max(z) + \
                        self._z_rand*self.p_rand(z))
        
        q = 100/(angle_step*np.abs(q))
        
        return q

    # precompute all possible rays to make ray cast table
    def pre_compute(self, occupancy_map, save_path="./ray_cast_table.npy"):
        from tqdm import tqdm

        h, w = occupancy_map.shape
        res = self._resolution
        num_angle = 360
        ray_table = np.zeros((h, w, num_angle), np.float64)

        for j in tqdm(range(0, h)):
            for i in range(0, w):

                b_t = np.array([i*res, j*res, 0.], np.float64)
                if occupancy_map[j, i] < 0:
                    continue

                for degree in range(0, num_angle):
                    rad = degree*np.pi/180
                    b_t[-1] = rad
                    ray_table[j, i, degree] = self.ray_casting(b_t)

        with open(save_path, 'wb') as f:
            np.save(f, ray_table)
        

    
