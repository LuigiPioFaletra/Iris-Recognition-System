import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import random

from config import segmentation
from utils import get_circle_means, get_mean_and_std

def find_areas(path):
    image = cv2.imread(path, 0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pupil = []
    multiplier, p_2 = segmentation["MULTIPLIER"], segmentation["PARAMS"]["PARAM_2"]
    while p_2 > segmentation["PUPIL"]["PARAM_2_LIMIT"] and len(pupil) < segmentation["PUPIL"]["LENGTH"]:
        for med, thr in [(m, t) for m in segmentation["PUPIL"]["MEDIAN"] for t in segmentation["PUPIL"]["THRESHOLD"]]:
            median = cv2.medianBlur(image, 2*med + 1)
            _, threshold = cv2.threshold(median, thr, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(threshold, contours, -1, (255), -1)
            edges = cv2.Canny(threshold, 20, 100)
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            ksize = 2 * random.randrange(5, 11) + 1
            edges = cv2.GaussianBlur(edges, (ksize, ksize), 0)
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 1, np.array([]), segmentation["PARAMS"]["PARAM_1"], p_2)
            if circles is not None and len(circles) > 0:
                circles = np.round(circles[0, :]).astype("int")
                pupil.extend(circles)
        p_2 -= 1
    cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    pupil = get_circle_means(pupil)
    if not pupil:
        return None
    radius = int(math.ceil(pupil[2] * 1.5))
    center = int(math.ceil(pupil[2] * multiplier)) 
    iris = find_iris(image, pupil, center, radius)
    while(not iris and multiplier <= 0.7):
        multiplier += 0.05
        center = int(math.ceil(pupil[2] * multiplier))
        iris = find_iris(image, pupil, center, radius)
    if not iris:
        return None
    detected_areas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.circle(detected_areas, (pupil[0], pupil[1]), pupil[2], (0, 0, 255), 1)
    cv2.circle(detected_areas, (iris[0], iris[1]), iris[2], (0, 255, 0), 1)
    if segmentation["USE_HOUGH_LINES_METHOD"]:
        edges = cv2.Canny(image, 50, 200, None, 3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 50, None, 50, 10)
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                if abs(theta - np.pi/2) < np.pi/18:
                    cv2.line(image, pt1, pt2, (0, 0, 0), 3, cv2.LINE_AA)
    else:
        edges = cv2.Canny(image, 50, 200, None, 3)
        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
        if linesP is not None:
            for i in range(len(linesP)):
                x1, y1, x2, y2 = linesP[i][0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 3, cv2.LINE_AA)
    if segmentation["SHOW"]["CONTOURS"]:
        cv2.imshow("Iris borders", detected_areas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    mask = image.copy()
    mask[:] = (0)
    cv2.circle(mask, (iris[0], iris[1]), iris[2], (255), -1)
    cv2.circle(mask, (pupil[0], pupil[1]), pupil[2], (0), -1)
    area = cv2.bitwise_and(image, mask)
    equalized_area = area.copy()
    cv2.equalizeHist(area, equalized_area)
    area = cv2.addWeighted(area, 0.0, equalized_area, 1.0, 0)
    if segmentation["SHOW"]["EQUALIZATION"]:
        cv2.imshow("Iris equalization", area)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    bg = area.copy()
    bg[:] = 0
    areas = {"image": bg.copy(), "pupil": pupil, "iris": iris, "keypoints": None,
             "init_keypoints": bg.copy(), "filtered_keypoints": bg.copy(), "description": None}
    for column in range(area.shape[1]):
        for row in range(area.shape[0]):
            if not math.sqrt((column-pupil[0])**2 + (row-pupil[1])**2) <= pupil[2] \
                and math.sqrt((column-iris[0])**2 + (row-iris[1])**2) <= iris[2]:
                areas["image"][row, column] = area[row, column]
    areas["iris"] = (int(1.25 * iris[2]), int(1.25 * iris[2]), int(iris[2]))
    x = areas["iris"][0] - iris[0]
    y = areas["iris"][1] - iris[1]
    areas["pupil"] = (int(x + pupil[0]), int(y + pupil[1]), int(pupil[2]))
    M = np.float32([[1, 0, x], [0, 1, y]])
    areas["image"] = cv2.warpAffine(areas["image"], M, (area.shape[1], area.shape[0]))
    areas["image"] = areas["image"][0:int(2.5 * iris[2]), 0:int(2.5 * iris[2])]
    sift = cv2.SIFT_create()
    areas["keypoints"] = sift.detect(areas["image"], None)
    areas["init_keypoints"] = cv2.drawKeypoints(areas["image"], areas["keypoints"], color=(0, 255, 0), flags=0, outImage=None)
    cv2.circle(areas["init_keypoints"], (areas["pupil"][0], areas["pupil"][1]), areas["pupil"][2], (0, 0, 255), 1)
    cv2.circle(areas["init_keypoints"], (areas["iris"][0], areas["iris"][1]), areas["iris"][2], (0, 255, 255), 1)
    filtered_keypoints = []
    for point in areas["keypoints"]:
        if not math.sqrt((point.pt[0] - areas["pupil"][0])**2 + (point.pt[1] - areas["pupil"][1])**2) <= areas["pupil"][2] + 3 and \
        math.sqrt((point.pt[0] - areas["iris"][0])**2 + (point.pt[1] - areas["iris"][1])**2) <= areas["iris"][2] - 5:
            filtered_keypoints.append(point)
    areas["keypoints"] = filtered_keypoints
    areas["filtered_keypoints"] = cv2.drawKeypoints(areas["image"], areas["keypoints"], color=(0, 255, 0), flags=0, outImage=None)
    cv2.circle(areas["filtered_keypoints"], (areas["pupil"][0], areas["pupil"][1]), areas["pupil"][2], (0, 0, 255), 1)
    cv2.circle(areas["filtered_keypoints"], (areas["iris"][0], areas["iris"][1]), areas["iris"][2], (0, 255, 255), 1)
    areas["keypoints"], areas["description"] = sift.compute(areas["image"], areas["keypoints"])
    if segmentation["SHOW"]["KEYPOINTS"]:
        plt.subplot(1, 2, 1), plt.imshow(areas["init_keypoints"])
        plt.title("Unfiltered iris"), plt.xticks([]), plt.yticks([])
        plt.subplot(1, 2, 2), plt.imshow(areas["filtered_keypoints"])
        plt.title("Filtered iris"), plt.xticks([]), plt.yticks([])
        plt.show()
    return areas

def find_iris(image, pupil, center, radius):
    def get_circles(hough_param, median_params, edge_params):
        total_circles = []
        for med, thr in [(m, t) for m in median_params for t in edge_params]:
            median = cv2.medianBlur(image, 2*med + 1)
            edges = cv2.Canny(median, 0, thr, apertureSize=5)
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            ksize = 2 * random.randrange(5, 11) + 1
            edges = cv2.GaussianBlur(edges, (ksize, ksize), 0)
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, minDist=20, param1=100, param2=hough_param)
            if circles is not None and circles.ndim == 3 and circles.shape[1] > 0:
                circles = np.round(circles[0, :]).astype("int")
                for (column, row, r) in circles:
                    if math.sqrt((column-int(pupil[0]))**2 + (row-int(pupil[1]))**2) <= center and r > radius:
                        total_circles.append((column, row, r))
        return total_circles
    total_circles = []
    p_2 = segmentation["PARAMS"]["PARAM_2"]
    while p_2 > segmentation["IRIS"]["PARAM_2_LIMIT"] and len(total_circles) < segmentation["CIRCLES_LENGTH"]:
        circles = get_circles(p_2, segmentation["IRIS"]["MEDIAN_1"], segmentation["IRIS"]["THRESHOLD"])
        if circles:
            total_circles += circles
        p_2 -= 1
    if not total_circles:
        p_2 = segmentation["PARAMS"]["PARAM_2"]
        while p_2 > segmentation["IRIS"]["PARAM_2_LIMIT"] and len(total_circles) < segmentation["CIRCLES_LENGTH"]:
            circles = get_circles(p_2, segmentation["IRIS"]["MEDIAN_2"], segmentation["IRIS"]["THRESHOLD"])
            if circles:
                total_circles += circles
            p_2 -= 1
    cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if not total_circles:
        return None
    mean_0, dev_0 = get_mean_and_std([int(i[0]) for i in total_circles])
    mean_1, dev_1 = get_mean_and_std([int(i[1]) for i in total_circles])
    filtered_circles, filtered_position, not_filtered_circles = [], [], []
    for c in total_circles[:]:
        if c[0] < mean_0 - segmentation["RATIO"]*dev_0 or c[0] > mean_0 + segmentation["RATIO"]*dev_0 \
            or c[1] < mean_1 - segmentation["RATIO"]*dev_1 or c[1] > mean_1 + segmentation["RATIO"]*dev_1:
            not_filtered_circles.append(c)
        else:
            filtered_position.append(c)
    if len([float(c[2]) for c in filtered_position]) < 3:
        filtered_circles = filtered_position
    else:
        alpha_circle, min_distance = None, None
        circles_1, circles_2 = filtered_position[:], filtered_position[:]
        for c_1 in circles_1:
            distance = 0
            for c_2 in circles_2:
                distance += math.fabs(float(c_1[2]) - float(c_2[2]))
            if not min_distance or distance < min_distance:
                min_distance = distance
                alpha_circle = c_1
        alpha_radius = alpha_circle[2]
        _, dev_radius = get_mean_and_std([float(c[2]) for c in filtered_position])
        max_radius = alpha_radius + dev_radius
        min_radius = alpha_radius - dev_radius
        for c in filtered_position:
            if c[2] < min_radius or c[2] > max_radius:
                not_filtered_circles.append(c)
            else:
                filtered_circles.append(c)
    return get_circle_means(filtered_circles)
