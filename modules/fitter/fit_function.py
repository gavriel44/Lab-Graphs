import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.odr import Model, ODR, RealData


class FitFunction:
    def __init__(self, func, initial_guesses=None):
        self.func = func
        self.default_guesses = initial_guesses

    def __call__(self, params, x):
        return self.func(params, x)

    def get_default_guesses(self):
        return self.default_guesses


class LinearFit(FitFunction):
    def __init__(self, initial_guesses=[1, 0]):
        super().__init__(self.linear_func, initial_guesses=initial_guesses)

    @staticmethod
    def linear_func(params, x):
        return params[1] * x + params[0]


class PolynomialFit(FitFunction):
    def __init__(self):
        super().__init__(self.polynomial_func, initial_guesses=[1, 1, 0])

    @staticmethod
    def polynomial_func(params, x):
        return params[2] * x ** 2 + params[1] * x + params[0]


class OpticsFit(FitFunction):
    def __init__(self):
        super().__init__(self.optics_func, initial_guesses=[1, 1])

    @staticmethod
    def optics_func(params, x):
        return params[1] * x / (x - params[1]) + params[0]


class ExponentialFit(FitFunction):
    def __init__(self):
        super().__init__(self.exponential_func, initial_guesses=[1, 1, 1])

    @staticmethod
    def exponential_func(params, x):
        return params[1] * np.exp(-(1 / params[2]) * x) + params[0]


class SinusoidalFit(FitFunction):
    def __init__(self, initial_guesses=[1, 1, 0, 1]):
        super().__init__(self.sinusoidal_func, initial_guesses=initial_guesses)

    @staticmethod
    def sinusoidal_func(params, x):
        return params[3] * np.sin(params[1] * x + params[2]) + params[0]


class BRMoodleFit(FitFunction):
    def __init__(self, initial_guesses=[1, 1, 1]):
        super().__init__(self.b_r_moodle_func, initial_guesses=initial_guesses)

    @staticmethod
    def b_r_moodle_func(params, x):
        return params[0] / ((x + params[1]) ** 3) + params[2]


class BRFit(FitFunction):
    def __init__(self):
        super().__init__(self.b_r_func, initial_guesses=[1, 1, 1, 1])

    @staticmethod
    def b_r_func(params, x):
        return params[0] / ((x + params[1]) ** params[2]) + params[3]


class BRZFit(FitFunction):
    def __init__(self, initial_guesses=[1, 1, 1, 1]):
        super().__init__(self.b_r_z_func, initial_guesses=initial_guesses)

    @staticmethod
    def b_r_z_func(params, x):
        return params[0] / (params[1] + ((x + params[2]) ** 2)) ** (3 / 2) + params[3]
