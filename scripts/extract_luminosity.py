import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Directory with your images
image_dir = 'C:/physics/lab/lab-data/Semester-B/electron-ecceleration/second-part-photos'

# A mapping of filenames to corresponding voltages
voltages = {
    '131.67-a.jpg': 131.67,
    '156.7-a.jpg': 156.7,
    '197.47-a.jpg': 197.47,
    '233.09-a.jpg': 233.09,
    '292.39-a.jpg': 292.39,
    '326.52-a.jpg': 326.52,
    # '339.62-a.jpg': 339.62,
    '387.12-a.jpg': 387.12,
    '420.34-a.jpg': 420.34,
    '433.07-a.jpg': 433.07,

    # add more...
}

brightness_data = []

for filename in voltages:
    filepath = os.path.join(image_dir, filename)
    img = cv2.imread(filepath)

    # Convert to HSV to isolate brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Option 1: Automatically find the bright green dot
    # You may tweak the ranges depending on your green dot's color
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Compute average brightness only where green mask is True
    # brightness = cv2.mean(hsv[:, :, 2], mask=mask)[0]  # V channel of HSV
    brightness = cv2.mean(hsv[:, :, 2])[0]  # V channel of HSV

    brightness_data.append((voltages[filename], brightness))

# Sort data by voltage
brightness_data.sort()


# # Show the original image and the mask side by side
# cv2.imshow('Original Image', img)
# cv2.imshow('Green Mask', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Plot
voltages, brightness = zip(*brightness_data)
plt.plot(voltages, brightness, 'o-')
plt.xlabel('Accelerating Voltage (V)')
plt.ylabel('Brightness (Luminosity)')
plt.title('CRT Brightness vs Voltage')
plt.grid(True)
plt.show()
