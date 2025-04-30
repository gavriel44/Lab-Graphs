import numpy as np
from scipy import stats
from scipy.odr import Model, ODR, RealData

from modules.fitter.fit_function import FitFunction
from utils.data_table import DataTable


class DataFitter:
    def __init__(self, data_table: DataTable, fit_func: FitFunction, initial_guesses=None):
        self.data_table = data_table
        self.fit_func = fit_func
        self.initial_guesses = initial_guesses or self.fit_func.get_default_guesses()

    def odr_fit(self):
        model = Model(self.fit_func)
        odr_data = RealData(self.data_table.x, self.data_table.y, sx=self.data_table.delta_x,
                            sy=self.data_table.delta_y)
        odr = ODR(data=odr_data, model=model, beta0=self.initial_guesses)
        output = odr.run()

        fit_params = output.beta
        fit_params_error = output.sd_beta
        fit_cov = output.cov_beta
        return fit_params, fit_params_error, fit_cov, output

    def fit(self):
        fit_params, fit_params_error, fit_cov, output = self.odr_fit()
        residuals, degrees_of_freedom, chi2_red, p_val = self.calc_stats(fit_params, output)

        return {
            "fit_params": fit_params,
            "fit_params_error": fit_params_error,
            "fit_cov": fit_cov,
            "residuals": residuals,
            "degrees_of_freedom": degrees_of_freedom,
            "chi2_red": chi2_red,
            "p_val": p_val,
            "output": output
        }

    def calc_stats(self, fit_params, output):
        x = self.data_table.x
        y = self.data_table.y
        fit_func = self.fit_func

        residuals = y - fit_func(fit_params, x)
        degrees_of_freedom = len(x) - len(fit_params)
        chi2 = output.sum_square
        chi2_red = chi2 / degrees_of_freedom
        p_val = stats.chi2.sf(chi2, degrees_of_freedom)
        return residuals, degrees_of_freedom, chi2_red, p_val

    def print_output(self, fit_params, fit_params_error, chi2_red, p_val, degrees_of_freedom):
        for i in range(len(fit_params)):
            print(
                f"a[{i}]: {fit_params[i]} /u00B1 {fit_params_error[i]} ({(abs(fit_params_error[i] / fit_params[i])) * 100}% error)")
        print(f"chi squared reduced = {chi2_red:.5f} /u00B1 {np.sqrt(2 / degrees_of_freedom)}")
        print(f"p-probability = {p_val:.5e}")
        print(f"DOF = {degrees_of_freedom}")
