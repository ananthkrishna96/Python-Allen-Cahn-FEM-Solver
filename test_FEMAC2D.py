#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 7 12:49:14 2023

@author: aorunnuk
"""

import numpy as np
from scipy.interpolate import PPoly, splrep
from scipy.special import roots_legendre
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from FEMAC2D import solver, plot_2d

def test_solver():
    eps = 0.01
    dt = 0.1
    ndofs = 30
    deg = 2

    func0 = lambda x, y: 1 / (1 + 100 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))  #Runge Function 2D

    eta_2d, basisfun = solver(func0, eps, dt, ndofs, deg)

    # Assert that the shape of eta_2d is correct
    assert eta_2d.shape == ((int(1 / dt) + 1), ndofs**2)

    assert np.max(eta_2d) < 1
    assert np.min(eta_2d) > 0

    print("Solver test passed.")

if __name__ == '__main__':
    test_solver()

