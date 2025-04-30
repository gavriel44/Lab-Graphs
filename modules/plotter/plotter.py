import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.odr import Model, ODR, RealData
from utils.data_table import DataTable

Y = 'ϵ_A [Volt]'
X = 'ω [rad/s]'

# Y = '(B_radial - B_earth)/μ_0 [A/m]'
# X = 'Θ [rad]'
MAIN_GRAPH_HEADING = f'{Y} as a function of {X}'
RESIDUALS_GRAPH_HEADING = f'Distance from fit as a function of {X}'


def format_output_str(fit_params, fit_params_error, chi2_red, p_val, degrees_of_freedom):
    re = ""
    for i in range(len(fit_params)):
        fit_i = str(fit_params[i])
        fit_error_i = str(fit_params_error[i])
        re += f"a[{i}]: {fit_i[:fit_i.index('.') + 4]} /u00B1 {fit_error_i[:fit_error_i.index('.') + 4]}/n"
    return re


class PlotterParams:
    def __init__(self, fit_func, fit_params, fit_params_error, chi2_red, p_val, degrees_of_freedom, residuals):
        self.fit_func = fit_func
        self.fit_params = fit_params
        self.fit_params_error = fit_params_error
        self.chi2_red = chi2_red
        self.p_val = p_val
        self.degrees_of_freedom = degrees_of_freedom
        self.residuals = residuals


class Plotter:
    def __init__(self, data_table: DataTable, plotter_params: PlotterParams = None):
        self.plotter_params = plotter_params
        self.data_table = data_table

    def plot(self):
        x = self.data_table.x
        delta_x = self.data_table.delta_x
        y = self.data_table.y
        delta_y = self.data_table.delta_y

        plt.close('all')
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        plt.style.use('classic')

        fig.patch.set_facecolor('white')
        for ax in axs:
            ax.set_facecolor('white')

        x_fit = np.linspace(min(x), max(x), 10 * len(x))
        y_fit = self.plotter_params.fit_func(self.plotter_params.fit_params, x_fit)
        axs[0].errorbar(x, y, xerr=delta_x, yerr=delta_y, fmt='.b', label='Data',
                        ecolor='gray')  # Change the label if needed

        fit_params = self.plotter_params.fit_params
        fit_params_error = self.plotter_params.fit_params_error
        chi2_red = self.plotter_params.chi2_red
        p_val = self.plotter_params.p_val
        degrees_of_freedom = self.plotter_params.degrees_of_freedom
        residuals = self.plotter_params.residuals

        axs[0].plot(x_fit, y_fit,
                    label=format_output_str(fit_params, fit_params_error, chi2_red, p_val, degrees_of_freedom),
                    c='r',
                    alpha=0.5)  # Change the label if needed

        # If you want to plot multiple functions, change here the relevant parameters (x, y, xerr, yerr, label). Otherwise, uncomment the 2 next lines:
        # axs[0].errorbar(x + 0.2, y + 0.3, xerr=delta_x, yerr=delta_y, fmt='.g', label='Data', ecolor='gray')
        # axs[0].plot(x_fit + 0.2, y_fit + 0.3, label='Fit', c='k', alpha=0.5)

        axs[0].set_title(self.get_main_graph_heading())  # Add here the full title for the fit
        axs[0].set_xlabel(self.data_table.get_x_label())  # Change x-axis label if needed
        axs[0].set_ylabel(self.data_table.get_y_label())  # Change y-axis label if needed

        axs[0].grid(True)
        # axs[0].legend(loc='upper left')

        axs[1].errorbar(x, residuals, xerr=delta_x, yerr=delta_y, fmt='.b', label="Data", ecolor='gray')
        axs[1].hlines(0, min(x), max(x), colors='r', linestyles='dashed')

        axs[1].set_title(self.get_residuals_graph_heading())  # Add here the full title for the residuals
        axs[1].set_xlabel(self.data_table.get_x_label())  # Change column names if needed
        axs[1].set_ylabel(
            f'{self.data_table.get_y_label()} - fit({self.data_table.get_x_label()})')  # Change column names if needed

        axs[1].grid(True)
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    def get_main_graph_heading(self):
        return f'{self.data_table.get_y_label()} as a function of {self.data_table.get_x_label()}'

    def get_residuals_graph_heading(self):
        return f'Distance from fit as a function of {self.data_table.get_x_label()}'
