import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.odr import Model, ODR, RealData

Y = 'ϵ_A [Volt]'
X = 'ω [rad/s]'

# Y = '(B_radial - B_earth)/μ_0 [A/m]'
# X = 'Θ [rad]'
MAIN_GRAPH_HEADING = f'{Y} as a function of {X}'
RESIDUALS_GRAPH_HEADING = f'Distance from fit as a function of {X}'


def get_excel(file_path, sheet_number=0):
    # file_path = r"/content/FreeFallGroupD.xls" # Replace with your file path: r"/contents/<your file name>.xlsx"
    return pd.read_excel(file_path, sheet_name=sheet_number)


def choose_columns(data, columns):
    x = data.iloc[:, columns[0]]
    delta_x = data.iloc[:, columns[1]]
    y = data.iloc[:, columns[2]]
    delta_y = data.iloc[:, columns[3]]
    return x, delta_x, y, delta_y


def function(A, x):
    pass  # Define your function. See some examples below.


def linear(A, x):
    return A[1] * x + A[0]


def polynomial(A, x):
    return A[2] * x ** 2 + A[1] * x + A[0]


def optics(A, x):
    return A[1] * x / (x - A[1]) + A[0]


def exponential(A, x):
    return A[1] * np.exp(-(1 / A[2]) * x) + A[0]


def sinusoidal(A, x):
    return A[3] * np.sin(A[1] * x + A[2]) + A[0]


def b_r_moodle(A, x):
    return A[0] / ((x + A[1]) ** 3) + A[2]


def b_r(A, x):
    return A[0] / ((x + A[1]) ** A[2]) + A[3]


def b_r_z(A, x):
    return A[0] / (A[1] + ((x + A[2]) ** 2)) ** (3 / 2) + A[2]


def odr_fit(fit_func, initial_guesses, x, delta_x, y, delta_y):
    model = Model(fit_func)
    odr_data = RealData(x, y, sx=delta_x, sy=delta_y)
    odr = ODR(data=odr_data, model=model, beta0=initial_guesses)
    output = odr.run()

    fit_params = output.beta
    fit_params_error = output.sd_beta
    fit_cov = output.cov_beta
    return fit_params, fit_params_error, fit_cov, output


def calc_stats(x, y, fit_func, fit_params, output):
    residuals = y - fit_func(fit_params, x)
    degrees_of_freedom = len(x) - len(fit_params)
    chi2 = output.sum_square
    chi2_red = chi2 / degrees_of_freedom
    p_val = stats.chi2.sf(chi2, degrees_of_freedom)
    return residuals, degrees_of_freedom, chi2_red, p_val


def format_output_str(fit_params, fit_params_error, chi2_red, p_val, degrees_of_freedom):
    re = ""
    for i in range(len(fit_params)):
        fit_i = str(fit_params[i])
        fit_error_i = str(fit_params_error[i])
        re += f"a[{i}]: {fit_i[:fit_i.index('.') + 4]} /u00B1 {fit_error_i[:fit_error_i.index('.') + 4]}/n"
    return re


def print_output(fit_params, fit_params_error, chi2_red, p_val, degrees_of_freedom):
    for i in range(len(fit_params)):
        print(
            f"a[{i}]: {fit_params[i]} /u00B1 {fit_params_error[i]} ({(abs(fit_params_error[i] / fit_params[i])) * 100}% error)")
    print(f"chi squared reduced = {chi2_red:.5f} /u00B1 {np.sqrt(2 / degrees_of_freedom)}")
    print(f"p-probability = {p_val:.5e}")
    print(f"DOF = {degrees_of_freedom}")


def show_plot(data):
    # columnsForPolinomal = [2, 3, 1, 0] # Define the columns indices to represent x, delta x, y, delta y.
    columnsForLinear = [0, 1, 2, 3]  # Define the columns indices to represent x, delta x, y, delta y. - for section A
    # columnsForLinear = [2, 3, 4, 5] # Define the columns indices to represent x, delta x, y, delta y. - for section B
    columns = columnsForLinear
    x, delta_x, y, delta_y = choose_columns(data, columns)

    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.style.use('classic')

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.errorbar(x, y, xerr=delta_x, yerr=delta_y, fmt='.b', label='Data', ecolor='gray')  # Change the label if needed

    ax.set_title('Data - Cycle Time(s) as a function of Length(m)')  # Add here the full title for the fit
    ax.set_xlabel(f'{data.columns[columns[0]]}')  # Change x-axis label if needed
    ax.set_ylabel(f'{data.columns[columns[2]]}')  # Change y-axis label if needed

    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()


def fit_and_show_plot(data, columns, x, delta_x, y, delta_y, fit_func):
    residuals, degrees_of_freedom, chi2_red, p_val, fit_params, fit_params_error, fit_cov, output = fit(fit_func, x,
                                                                                                        delta_x, y,
                                                                                                        delta_y)

    plt.close('all')
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    plt.style.use('classic')

    fig.patch.set_facecolor('white')
    for ax in axs:
        ax.set_facecolor('white')

    x_fit = np.linspace(min(x), max(x), 10 * len(x))
    y_fit = fit_func(fit_params, x_fit)
    axs[0].errorbar(x, y, xerr=delta_x, yerr=delta_y, fmt='.b', label='Data',
                    ecolor='gray')  # Change the label if needed
    axs[0].plot(x_fit, y_fit,
                label=format_output_str(fit_params, fit_params_error, chi2_red, p_val, degrees_of_freedom), c='r',
                alpha=0.5)  # Change the label if needed

    # If you want to plot multiple functions, change here the relevant parameters (x, y, xerr, yerr, label). Otherwise, uncomment the 2 next lines:
    # axs[0].errorbar(x + 0.2, y + 0.3, xerr=delta_x, yerr=delta_y, fmt='.g', label='Data', ecolor='gray')
    # axs[0].plot(x_fit + 0.2, y_fit + 0.3, label='Fit', c='k', alpha=0.5)

    axs[0].set_title(MAIN_GRAPH_HEADING)  # Add here the full title for the fit
    axs[0].set_xlabel(f'{data.columns[columns[0]]}')  # Change x-axis label if needed
    axs[0].set_ylabel(f'{data.columns[columns[2]]}')  # Change y-axis label if needed

    axs[0].grid(True)
    # axs[0].legend(loc='upper left')

    axs[1].errorbar(x, residuals, xerr=delta_x, yerr=delta_y, fmt='.b', label="Data", ecolor='gray')
    axs[1].hlines(0, min(x), max(x), colors='r', linestyles='dashed')

    axs[1].set_title(RESIDUALS_GRAPH_HEADING)  # Add here the full title for the residuals
    axs[1].set_xlabel(f'{data.columns[columns[0]]}')  # Change column names if needed
    axs[1].set_ylabel(f'{data.columns[columns[2]]} - fit({data.columns[columns[0]]})')  # Change column names if needed

    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def fit(fit_func, x, delta_x, y, delta_y, print_res=True):
    initial_guesses = (1, 1)
    # initial_guesses = (45000, 100, 1.7, 80000)

    # def sinusoidal(A, x):
    #     return A[3] * np.sin(A[1] * x + A[2]) + A[0]

    # initial_guesses = (0.2,
    #                    5.1, 1,
    #                    0.01)  # Define the initial guesses for the parameters in list "A" (make sure they are the same length, and in the same order!)
    fit_params, fit_params_error, fit_cov, output = odr_fit(fit_func, initial_guesses, x, delta_x, y, delta_y)
    residuals, degrees_of_freedom, chi2_red, p_val = calc_stats(x, y, fit_func, fit_params, output)

    if print_res:
        print_output(fit_params, fit_params_error, chi2_red, p_val, degrees_of_freedom)

    return residuals, degrees_of_freedom, chi2_red, p_val, fit_params, fit_params_error, fit_cov, output


def main():
    print("Start")
    # file_path = r"C:/physics/lab/lab-data/optics-21/data.xlsx"
    # file_path = r"C:/physics/lab/lab-data/Semester-B/magnetism/Part2.xlsx"
    file_path = r"C:/physics/lab/lab-data/Semester-B/magnetism/Part1-local-gavri-comp-2.xlsx"

    sheet = get_excel(file_path)
    # sheet.insert(1, "dTime ( s )", 0.05 / np.sqrt(12))
    # sheet.insert(3, "dPosition ( m )", 0.001 / np.sqrt(12))
    # sheet = sheet.iloc[1:-20]

    print(sheet)

    columnsForLinear = [0, 1, 2, 3]  # Define the columns indices to represent x, delta x, y, delta y.
    # columnsForLinear = [8, 9, 6, 7]
    # columnsForLinear.reverse()

    columns = columnsForLinear
    x, delta_x, y, delta_y = choose_columns(sheet, columns)

    # show_plot(sheet)
    # fit(sinusoidal, x, delta_x, y, delta_y)

    # show_plot(data=sheet)

    fit_and_show_plot(sheet, columns, x, delta_x, y, delta_y, linear)


if __name__ == '__main__':
    main()

#
# columnsForLinear = [0, 1, 2, 3]  # Define the columns indices to represent x, delta x, y, delta y.
# columns = columnsForLinear
# x, delta_x, y, delta_y = choose_columns(sheet, columns)
#
# show_plot(sheet)
# fit(sinusoidal, x, delta_x, y, delta_y)
#
# fit_and_show_plot(sheet, columns, x, delta_x, y, delta_y, sinusoidal)

# x_values = np.arange(0, 8 * np.pi, 0.1)
#
# # Compute sin(x) for each x
# y_values = np.sin(x_values) + 2
#
# # Create a DataFrame
# sheet = pd.DataFrame({
#     "x": x_values,
#     "sin(x)": y_values
# })
#
# sheet.insert(1, "dTime ( s )", 0.05 / np.sqrt(12))
# sheet.insert(3, "dPosition ( m )", 0.001 / np.sqrt(12))
#
# print(sheet)
