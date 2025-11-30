import numpy as np
import os

from config import main
from segmentation import find_areas
from utils import get_matches, load_from_pkl, plot_roc, save_to_pkl, show_curve, show_FRR_and_FAR_results

def compare_images(path_1, path_2):
    path_1_pkl = path_1.replace(main["EXTENSION"], main["DATA"])
    path_2_pkl = path_2.replace(main["EXTENSION"], main["DATA"])
    if os.path.exists(path_1_pkl):
        print(f"Loading saved data for the image to the path '{path_1}'")
        areas_1 = load_from_pkl(path_1_pkl)
    else:
        print(f"Image analysis at path '{path_1}'")
        areas_1 = find_areas(path_1)
        save_to_pkl(areas_1, path_1_pkl)
    if os.path.exists(path_2_pkl):
        print(f"Loading saved data for the image to the path '{path_2}'")
        areas_2 = load_from_pkl(path_2_pkl)
    else:
        print(f"Image analysis at path '{path_2}'")
        areas_2 = find_areas(path_2)
        save_to_pkl(areas_2, path_2_pkl)
    matches = get_matches(areas_1, areas_2, main["DEV_RATIO"], main["DEV_ANGLE"], main["DEV_DISTANCE"])
    return matches

def find_optimal_threshold_and_error_metrics(intra_class, inter_class, y_true, y_score):
    threshold, max_false_rejections_number, max_false_acceptance_number, max_sum = 0, 0, 0, float("inf")
    false_rejections_list, false_acceptance_list, FRR_list, FAR_list = [], [], [], []
    max_value = max(y_score)
    thresholds = np.linspace(1, max_value, max_value).astype(int)
    for thr in thresholds:
        false_rejections_number, false_acceptance_number = 0, 0
        y_score_boolean = [value >= thr for value in y_score]
        for val_1, val_2 in zip(y_score_boolean, y_true):
            if val_1 == False and val_2 == True:
                false_rejections_number += 1
            elif val_1 == True and val_2 == False:
                false_acceptance_number += 1
        sum = false_rejections_number + false_acceptance_number
        false_rejections_list.append(false_rejections_number)
        false_acceptance_list.append(false_acceptance_number)
        if sum < max_sum:
            threshold = thr
            max_false_rejections_number = false_rejections_number
            max_false_acceptance_number = false_acceptance_number
            max_sum = sum
        FRR, FAR = show_FRR_and_FAR_results(threshold, max_false_rejections_number, max_false_acceptance_number, intra_class, inter_class)
        FRR_list.append(FRR)
        FAR_list.append(FAR)
    return thresholds, false_rejections_list, false_acceptance_list

def load_and_compare_dataset_images():
    intra_class, inter_class = 0, 0
    y_true, y_score = [], []
    compared_pairs = set()
    subject_folders = [f for f in os.listdir(main["DATASET"]) if os.path.isdir(os.path.join(main["DATASET"], f))]
    for subject_folder in subject_folders:
        subject_path = os.path.join(main["DATASET"], subject_folder)
        images = [os.path.join(subject_path, subfolder, f)
                  for subfolder in os.listdir(subject_path)
                  if os.path.isdir(os.path.join(subject_path, subfolder))
                  for f in os.listdir(os.path.join(subject_path, subfolder))
                  if f.endswith(main["EXTENSION"])]
        for i, img1 in enumerate(images):
            for img2 in images[i + 1:]:
                pair = (img1, img2) if img1 < img2 else (img2, img1)
                if pair not in compared_pairs:
                    compared_pairs.add(pair)
                    matches = compare_images(img1, img2)
                    y_score.append(matches)
                    intra_class += 1
                    y_true.append(True)
        for other_subject_folder in subject_folders:
            if other_subject_folder != subject_folder:
                other_subject_path = os.path.join(main["DATASET"], other_subject_folder)
                other_images = [os.path.join(other_subject_path, subfolder, f)
                                for subfolder in os.listdir(other_subject_path)
                                if os.path.isdir(os.path.join(other_subject_path, subfolder))
                                for f in os.listdir(os.path.join(other_subject_path, subfolder))
                                if f.endswith(main["EXTENSION"])]
                for img1 in images:
                    for img2 in other_images:
                        pair = (img1, img2) if img1 < img2 else (img2, img1)
                        if pair not in compared_pairs:
                            compared_pairs.add(pair)
                            matches = compare_images(img1, img2)
                            y_score.append(matches)
                            inter_class += 1
                            y_true.append(False)
    return intra_class, inter_class, y_true, y_score


if __name__ == "__main__":
    intra_class, inter_class, y_true, y_score = load_and_compare_dataset_images()
    thresholds, false_rejections_list, false_acceptance_list = \
    find_optimal_threshold_and_error_metrics(intra_class, inter_class, y_true, y_score)
    if main["SHOW_REJECTIONS_CURVE"]:
        show_curve(thresholds, false_rejections_list, "rejections")
    if main["SHOW_ACCEPTANCES_CURVE"]:
        show_curve(thresholds, false_acceptance_list, "acceptances")
    if main["SHOW_ROC_CURVE"]:
        plot_roc(y_true, y_score)
