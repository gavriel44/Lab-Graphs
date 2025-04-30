import numpy as np


def print_fit_output(fit_results):
    fit_params = fit_results["fit_params"]
    fit_params_error = fit_results["fit_params_error"]
    chi2_red = fit_results["chi2_red"]
    p_val = fit_results["p_val"]
    degrees_of_freedom = fit_results["degrees_of_freedom"]

    for i in range(len(fit_params)):
        param = fit_params[i]
        error = fit_params_error[i]
        percent_error = abs(error / param) * 100 if param != 0 else float('inf')
        print(f"a[{i}]: {param} ± {error} ({percent_error}% error)")

    chi2_error = np.sqrt(2 / degrees_of_freedom) if degrees_of_freedom != 0 else float('inf')
    print(f"chi squared reduced = {chi2_red} ± {chi2_error}")
    print(f"p-probability = {p_val}")
    print(f"DOF = {degrees_of_freedom}")
