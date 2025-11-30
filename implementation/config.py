main = {
    "DATA": "",
    "DATASET": "./CASIA (version 1.0)",
    "DEV_ANGLE": 10,
    "DEV_DISTANCE": 0.15,
    "DEV_RATIO": 0.8,
    "EXTENSION": ".bmp",
    "SHOW_ACCEPTANCES_CURVE": True,
    "SHOW_REJECTIONS_CURVE": True,
    "SHOW_ROC_CURVE": True
}

segmentation = {
    "CIRCLES_LENGTH": 50,
    "IRIS": {
        "MEDIAN_1": [8, 10, 12, 14, 16, 18, 20],
        "MEDIAN_2": [3, 5, 7, 21, 23, 25],
        "PARAM_2_LIMIT": 40,
        "THRESHOLD": [430, 480, 530]
    },
    "MULTIPLIER": 0.25,
    "PARAMS": {
        "PARAM_1": 200,
        "PARAM_2": 120,
    },
    "PUPIL": {
        "LENGTH": 100,
        "MEDIAN": [3, 5, 7],
        "PARAM_2_LIMIT": 35,
        "THRESHOLD": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    },
    "RATIO": 1.5,
    "SHOW": {
        "CONTOURS": False,
        "EQUALIZATION": False,
        "KEYPOINTS": False
    },
    "USE_HOUGH_LINES_METHOD": True
}

utils = {
    "DATA": "",
    "KEYPOINTS": "",
    "SHOW_MATCHES": False
}

match segmentation["USE_HOUGH_LINES_METHOD"]:
    case True:
        main["DATA"] = "_HoughLines_data.pkl"
        utils["DATA"] = main["DATA"]
        utils["KEYPOINTS"] = "_HoughLines_keypoints.txt"
    case False:
        main["DATA"] = "_HoughLinesP_data.pkl"
        utils["DATA"] = main["DATA"]
        utils["KEYPOINTS"] = "_HoughLinesP_keypoints.txt"
