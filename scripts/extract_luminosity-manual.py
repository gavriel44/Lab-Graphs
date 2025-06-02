import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from modules.fitter.data_fitter import DataFitter
from modules.fitter.fit_function import ExponentialFit, LinearFit
from modules.plotter.plotter import PlotterParams, Plotter
from utils.data_table import DataTable
from utils.printing import print_fit_output

# --------------- Extract luminosity from images ---------------
# This script extracts luminosity from images of a cathode ray tube experiment.
# It allows the user to click on a point in the image, and it calculates the average brightness in a region around that point.
# It then saves the results to an Excel file and plots the data with a fitted curve.

image_dir = 'C:/physics/lab/lab-data/Semester-B/electron-ecceleration/second-part-photos'  # Replace with your folder
# ROI_SIZE = 45  # Region size (in pixels) around your click
ROI_SIZE = 60  # Region size (in pixels) around your click
SCALE_PERCENT = 60  # Resize images for easier display
RES_ERROR_BRIGHTNESS = (1 / 265) / np.sqrt(12)
BRIGHTNESS_VALUE_ERROR = np.sqrt(
    3) * RES_ERROR_BRIGHTNESS  # Error in brightness measurement (assumed uniform distribution)

TOTAL_PIXELS = 11287
TOTAL_BRIGHTNESS_ERROR = np.sqrt(TOTAL_PIXELS) * BRIGHTNESS_VALUE_ERROR  # Error in total brightness measurement


def get_voltage_res_error(voltage):
    return (voltage * (0.1 / 100)) + 0.05


# --- Image filenames and voltages ---
# voltages = {
#     '131.67-a.jpg': 131.67,
#     '156.7-a.jpg': 156.7,
#     '197.47-a.jpg': 197.47,
#     '233.09-a.jpg': 233.09,
#     '292.39-a.jpg': 292.39,
#     '326.52-a.jpg': 326.52,
#     '339.62-a.jpg': 339.62,
#     '387.12-a.jpg': 387.12,
#     '420.34-a.jpg': 420.34,
#     '433.07-a.jpg': 433.07,
#
#     # add more...
# }

voltages = {
    '131.67-a.jpg': 131.67,
    '156.7-a.jpg': 156.7,
    '197.47-a.jpg': 197.47,
    '233.09-a.jpg': 233.09,
    '292.39-a.jpg': 292.39,
    '326.52-a.jpg': 326.52,
    '339.62-a.jpg': 339.62,
    '387.12-a.jpg': 387.12,
    '420.34-a.jpg': 420.34,
    '433.07-a.jpg': 433.07,

    # add more...
}

# --- Store results ---
brightness_data = []


# --- Click handler ---
def click_and_measure(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img = param['image']
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # img = param['image']
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # luminance_roi = yuv[y1:y2, x1:x2, 0]

        b, g, r = cv2.split(img)  # OpenCV loads in BGR order
        raw_brightness = b.astype(np.uint16) + g.astype(np.uint16) + r.astype(np.uint16)

        # Get small region around click (Value channel)
        x1 = max(x - ROI_SIZE, 0)
        y1 = max(y - ROI_SIZE, 0)
        x2 = min(x + ROI_SIZE, img.shape[1])
        y2 = min(y + ROI_SIZE, img.shape[0])
        # roi = yuv[y1:y2, x1:x2, 0]
        roi = raw_brightness[y1:y2, x1:x2]

        # roi is a square region of shape (h, w)
        h, w = roi.shape
        center = (h // 2, w // 2)

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)

        radius = min(h, w) // 2
        circular_mask = dist_from_center <= radius

        # print amount of pixels in the circular mask
        print(f'Circular mask contains {np.sum(circular_mask)} pixels')

        # Apply circular mask
        circular_pixels = roi[circular_mask]

        # print(img.dtype)  # should be uint8
        # print(np.min(img), np.max(img))  # should be 0–255
        # print(np.min(circular_pixels), np.max(circular_pixels))

        # Plot 3D graph

        X2, Y2 = np.meshgrid(np.arange(w), np.arange(h))

        x_coords = X2[circular_mask]
        y_coords = Y2[circular_mask]
        z_values = circular_pixels

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_coords, y_coords, z_values, c=z_values, cmap='viridis', s=5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Brightness (Z)')
        ax.set_title(f'3D Brightness Map - Voltage: {param["voltage"]} V')
        plt.show()

        # avg_brightness = np.mean(circular_pixels)
        circular_pixels_floats = circular_pixels.astype(np.float32)
        brightness_sum = np.sum(circular_pixels_floats)

        # avg_brightness = len([x for x in circular_pixels if x == 255])  # Filter out zero values
        # std_brightness = np.std(circular_pixels)
        print(f'Image: {param["filename"]}, Voltage: {param["voltage"]}, Brightness: {brightness_sum:.2f}')

        brightness_data.append(
            (param['voltage'], get_voltage_res_error(param['voltage']), brightness_sum, TOTAL_BRIGHTNESS_ERROR))
        cv2.destroyAllWindows()  # Close the image window


# --- Main loop over images ---
for filename in voltages:
    filepath = os.path.join(image_dir, filename)
    img = cv2.imread(filepath)

    # Resize for display
    width = int(img.shape[1] * SCALE_PERCENT / 100)
    height = int(img.shape[0] * SCALE_PERCENT / 100)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    # print(img.dtype)  # should be uint8
    # print(np.min(img), np.max(img))  # should be 0–255
    # # print(np.min(luminance), np.max(luminance))

    # Show image and wait for click
    print(f"/nClick on the green dot in: {filename} (Voltage: {voltages[filename]} V)")
    cv2.imshow("Click the green dot", img)
    cv2.setMouseCallback("Click the green dot", click_and_measure, {
        'image': img,
        'filename': filename,
        'voltage': voltages[filename]
    })
    cv2.waitKey(0)

column_indexes = [0, 1, 2, 3]
data_table = DataTable.from_list(brightness_data, main_columns_indxs=column_indexes,
                                 column_names=["Voltage (V)", "Voltage Error (V)", "Brightness", "Brightness Error"])
# data_table = DataTable.from_excel(file_path="C:/physics/lab/lab-data/Semester-B/electron-ecceleration/output.xlsx")

# data_table.data_table.to_excel("C:/physics/lab/lab-data/Semester-B/electron-ecceleration/output-max-brightness.xlsx",
#                                index=False)  # Windows

# fit_function = ExponentialFit(initial_guesses=[1, 3.089, 0.007])
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
