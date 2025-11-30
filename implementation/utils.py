import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle

from config import utils
from sklearn.metrics import auc, roc_curve

def get_circle_means(circles):
    mean_0 = int(np.mean([c[0] for c in circles]))
    mean_1 = int(np.mean([c[1] for c in circles]))
    mean_2 = int(np.mean([c[2] for c in circles]))
    return mean_0, mean_1, mean_2

def get_matches(areas_1, areas_2, dev_ratio, dev_angle, dev_dist):
    matches = []
    number_of_matches = 0
    if not areas_1["keypoints"] or not areas_2["keypoints"]:
        return None
    else:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(areas_1["description"], areas_2["description"], k=2)
        points_1 = areas_1["keypoints"]
        points_2 = areas_2["keypoints"]
        distance_1_1 = areas_1["iris"][2] - areas_1["pupil"][2]
        distance_2_1 = areas_2["iris"][2] - areas_2["pupil"][2]
        angles, distances, filtered_circles = [], [], []
        for m, n in matches:
            if (m.distance/n.distance) > dev_ratio:
                continue
            x1, y1 = points_1[m.queryIdx].pt
            x2, y2 = points_2[m.trainIdx].pt
            angle_1 = math.degrees(math.atan2(-(areas_1["pupil"][1]-y1), (areas_1["pupil"][0]-x1)))
            angle_2 = math.degrees(math.atan2(-(areas_2["pupil"][1]-y2), (areas_2["pupil"][0]-x2)))
            angle = angle_1 - angle_2
            angles.append(angle)
            distance_1_2 = math.sqrt((areas_1["pupil"][0]-x1)**2 + (areas_1["pupil"][1]-y1)**2)
            distance_1_2 = distance_1_2 - areas_1["pupil"][2]
            distance_1_2 = distance_1_2 / distance_1_1
            distance_2_2 = math.sqrt((areas_2["pupil"][0]-x2)**2 + (areas_2["pupil"][1]-y2)**2)
            distance_2_2 = distance_2_2 - areas_2["pupil"][2]
            distance_2_2 = distance_2_2 / distance_2_1
            distance = distance_1_2 - distance_2_2
            distances.append(distance)
            filtered_circles.append(m)
        if filtered_circles:
            median_angle = np.median(np.array(angles))
            median_distance = np.median(np.array(distances))
            for m in filtered_circles[:]:
                x1, y1 = points_1[m.queryIdx].pt
                x2, y2 = points_2[m.trainIdx].pt
                angle_1 = math.degrees(math.atan2(-(areas_1["pupil"][1]-y1), (areas_1["pupil"][0]-x1)))
                angle_2 = math.degrees(math.atan2(-(areas_2["pupil"][1]-y2), (areas_2["pupil"][0]-x2)))
                angle = angle_1 - angle_2
                good_angle = (angle > median_angle - dev_angle and angle < median_angle + dev_angle)
                distance_1_2 = math.sqrt((areas_1["pupil"][0]-x1)**2 + (areas_1["pupil"][1]-y1)**2)
                distance_1_2 = distance_1_2 - areas_1["pupil"][2]
                distance_1_2 = distance_1_2 / distance_1_1
                distance_2_2 = math.sqrt((areas_2["pupil"][0]-x2)**2 + (areas_2["pupil"][1]-y2)**2)
                distance_2_2 = distance_2_2 - areas_2["pupil"][2]
                distance_2_2 = distance_2_2 / distance_2_1
                distance = distance_1_2 - distance_2_2
                good_distance = (distance > median_distance - dev_dist and distance < median_distance + dev_dist)
                if good_angle and good_distance:
                    continue
                filtered_circles.remove(m)
        matches = filtered_circles
        number_of_matches = len(matches)
    print(f"Common keypoints: {len(matches)}")
    image = cv2.drawMatchesKnn(areas_1["image"], areas_1["keypoints"], areas_2["image"],
                               areas_2["keypoints"], [matches], flags=2, outImg=None)
    matches.append(image)
    if utils["SHOW_MATCHES"]:
        cv2.imshow("Keypoints comparison", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return number_of_matches

def get_mean_and_std(x):
    sum, sum_squared = 0.0, 0.0
    for i in range(len(x)):
        sum += x[i]
    m = sum/len(x)
    for i in range(len(x)):
        sum_squared += (x[i]-m)**2
    return m, math.sqrt(sum_squared/len(x))

def load_from_pkl(filename):
    data = {}
    with open(filename, "rb") as f:
        data = pickle.load(f)
    try:
        with open(filename.replace(utils["DATA"], utils["KEYPOINTS"]), "r") as f:
            keypoints = []
            for line in f:
                parts = line.strip().split(", ")
                point = cv2.KeyPoint(float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]),
                                     float(parts[4]), int(parts[5]), int(parts[6]))
                keypoints.append(point)
            data["keypoints"] = keypoints
    except FileNotFoundError:
        pass
    return data

def plot_roc(FPR, TPR):
    fpr, tpr, _ = roc_curve(FPR, TPR)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Accept Rate")
    plt.ylabel("False Reject Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()

def save_to_pkl(data, filename):
    filtered_data = {key: value for key, value in data.items() if key != "keypoints"}
    with open(filename, "wb") as f:
        pickle.dump(filtered_data, f)
    f = open(filename.replace(utils["DATA"], utils["KEYPOINTS"]), "w")
    for point in data["keypoints"]:
        p = str(point.pt[0]) + ", " + str(point.pt[1]) + ", " + str(point.size) + ", " + str(point.angle) + ", " \
        + str(point.response) + ", " + str(point.octave) + ", " + str(point.class_id) + "\n"
        f.write(p)
    f.close()

def show_curve(thresholds, list, label):
    plt.figure()
    plt.plot(thresholds, list, label=label, color="blue")
    plt.xlabel("Threshold")
    plt.ylabel(f"False {label} number")
    plt.title(f"False {label} curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def show_FRR_and_FAR_results(threshold, false_rejections, false_acceptances, FRR_comparisons, FAR_comparisons):
    FRR = false_rejections / FRR_comparisons
    FAR = false_acceptances / FAR_comparisons
    print(f"\nBest threshold: {threshold}")
    print(f"False rejections number: {false_rejections}")
    print(f"Total intra-class comparisons: {FRR_comparisons}")
    print(f"False acceptances number: {false_acceptances}")
    print(f"Total inter-class comparisons: {FAR_comparisons}")
    print(f"FRR: {FRR:.4f}")
    print(f"FRR %: {FRR * 100:.2f}%")
    print(f"FAR: {FAR:.4f}")
    print(f"FAR %: {FAR * 100:.2f}%")
    return FRR, FAR
