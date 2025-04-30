from modules.fitter.data_fitter import DataFitter
from modules.fitter.fit_function import LinearFit, BRMoodleFit, SinusoidalFit, BRZFit
from modules.plotter.plotter import Plotter, PlotterParams
from utils.data_table import DataTable
from utils.printing import print_fit_output


def main():
    # file_path = r"C:/physics/lab/lab-data/Semester-B/magnetism/Part1-local-gavri-comp-2.xlsx"
    file_path = r"C:/physics/lab/lab-data/Semester-B/magnetism/Part2.xlsx"
    column_indexes = [0, 1, 2, 3]
    data_table = DataTable(file_path=file_path, main_columns_indxs=column_indexes)
    # fit_function = BRMoodleFit(initial_guesses=[0, 0, 1])
    # fit_function = BRZFit(initial_guesses=[2, 1, 0, 0])
    fit_function = LinearFit()

    fitter = DataFitter(data_table, fit_function)
    fit_results = fitter.fit()

    print_fit_output(fit_results)

    plotter_params = PlotterParams(
        fit_func=fit_function,
        fit_params=fit_results["fit_params"],
        fit_params_error=fit_results["fit_params_error"],
        chi2_red=fit_results["chi2_red"],
        p_val=fit_results["p_val"],
        degrees_of_freedom=fit_results["degrees_of_freedom"],
        residuals=fit_results["residuals"]
    )

    plotter = Plotter(data_table, plotter_params=plotter_params)
    plotter.plot()


if __name__ == '__main__':
    main()
