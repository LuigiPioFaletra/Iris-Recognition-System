# Python Project – Iris Recognition System (Spatial Domain)

## Overview

This repository contains a Python-based **iris recognition system** that operates in the **spatial domain** using keypoint extraction and comparison techniques.  
The project implements:

- Iris and pupil segmentation  
- Keypoint detection using SIFT  
- Keypoint matching with filtering based on angle, distance, and ratio metrics  
- FRR (False Reject Rate) and FAR (False Accept Rate) evaluation  
- ROC curve generation  
- Automated intra-class and inter-class comparisons  
- Configurable parameters via a dedicated configuration file  

The system was developed as part of coursework in **Biometrics / Image Processing**, and tested on Visual Studio Code.

---

## Repository Structure

```
main_repository/
│
├── implementation/
│   ├── config.py
│   ├── main.py
│   ├── segmentation.py
│   └── utils.py
│
├── description.pdf
├── LICENSE
└── README.md
```

---

## Introduction

### Scope of the System
The iris—an annular region situated between the pupil and sclera—contains a highly distinctive texture. This system implements a **unimodal iris recognition pipeline**, leveraging iris keypoints to determine identity similarities between images.

The primary objective is to provide a **robust, accurate, and efficient** biometric identification system that remains reliable under:

- lighting variations  
- eye tilt  
- partial occlusions  
- limited computational resources  

### Context
The project focuses on **keypoint-based biometric comparison**, an approach that:

- extracts reference points describing the iris texture  
- minimizes variations due to external conditions  
- offers higher robustness than many traditional biometric methods  

This methodology is particularly useful in **access control**, **security**, and **high-accuracy identity verification** applications.

### Importance of Iris-Based Recognition
Iris recognition through keypoints is one of the most reliable biometric identification methods, due to:

- low dependency on image resolution  
- fast and precise matching  
- strong resistance to spoofing  
- minimal false positives/negatives  

Applications include:

- biometric authentication  
- airport and secure facility access  
- digital identity verification systems  

---

## System Description

### General Architecture
The system is composed of five stages:

1. **Dataset loading** (CASIA v1.0)  
2. **Preprocessing** (filters, enhancement, equalization)  
3. **Segmentation** (iris/pupil localization via Hough transforms)  
4. **Keypoint extraction** (SIFT, contour masking, eyelash removal)  
5. **Matching and decision-making** (similarity scoring, thresholding)

---

## File Structure and Description

### `config.py`
Contains all configuration parameters for:

#### Main settings
- `DATA` – output file name  
- `DATASET` – path to CASIA dataset  
- `DEV_ANGLE`, `DEV_DISTANCE`, `DEV_RATIO` – matching tolerances  
- `EXTENSION` – image extension (`.bmp`)  
- Toggle plots:
  - `SHOW_ACCEPTANCES_CURVE`  
  - `SHOW_REJECTIONS_CURVE`  
  - `SHOW_ROC_CURVE`

#### Segmentation settings
- `CIRCLES_LENGTH` – max circle count  
- `IRIS` – thresholds, median filters, parameter limits  
- `PUPIL` – similar configuration for pupil detection  
- `RATIO` – iris/pupil geometric ratio  
- `SHOW` – debug displays:
  - contours  
  - equalization  
  - keypoints  
- `USE_HOUGH_LINES_METHOD` – choose between:
  - `HoughLines` (global lines)
  - `HoughLinesP` (probabilistic, segmented)

#### Utils settings
- File handling for `.pkl` and `.txt`  
- Display options for keypoint matches  

---

### `main.py`

The main executive script.  
Implements:

#### `compare_images()`
- Loads cached `.pkl` & `.txt` data if available  
- Otherwise extracts fresh segmentation + keypoints  
- Computes number of shared keypoints  
- Saves processed outputs  

#### `find_optimal_threshold_and_error_metrics()`
- Iterates across thresholds  
- Computes:
  - FRR (False Reject Rate)  
  - FAR (False Accept Rate)  
- Finds threshold minimizing FRR + FAR  
- Generates ROC curve (if enabled)

#### `load_and_compare_dataset_images()`
- Iterates through CASIA dataset folders  
- Performs:
  - intra-class comparisons  
  - inter-class comparisons  
- Saves all match counts  
- Computes FRR and FAR  

---

### `segmentation.py`

Contains iris/pupil localization and preprocessing functions, using **OpenCV**.

#### `find_areas()`
- Loads grayscale image  
- Detects pupil (median filters + thresholds + Hough)  
- Detects iris  
- Optional debug overlays  
- Performs equalization  
- Extracts SIFT keypoints  
- Filters out:
  - inner pupil points  
  - outer iris boundary points  
  - eyelash regions  

#### `find_iris()`
- Targets iris detection alone  
- Uses Hough transforms  
- Refines candidates via statistical validation  
- Returns averaged iris parameters  

---

### `utils.py`

Auxiliary functions for data handling and visualization.

#### Key functions include:
- `get_circle_means()` – averages circle parameters  
- `get_matches()` – SIFT BFMatcher + advanced filtering  
- `get_mean_and_std()` – statistics utilities  
- `load_from_pkl()` – loads image data  
- `save_to_pkl()` – saves data + keypoints (text format)  
- `plot_roc()` – generates ROC curve + AUC  
- `show_curve()` – plots false accept/reject trends  
- `show_FRR_and_FAR_results()` – displays summary metrics  

---

## Results and Performance

The system was evaluated through **intra-class** and **inter-class** comparisons using:

- `HoughLines`  
- `HoughLinesP`  

| Technique        | Optimal Threshold | False Rejections | Total Intra | False Acceptances | Total Inter   | FRR (%) | FAR (%) |
|------------------|------------------|------------------|--------------|--------------------|----------------|---------|---------|
| HoughLines       | 10               | 134              | 2268         | 28                 | 283122         | 5.91%   | 0.01%   |
| HoughLinesP      | 10               | 225              | 2268         | 28                 | 283122         | 9.92%   | 0.01%   |

Additional visual outputs generated by the system:

- ROC curves  
- False rejection curves  
- False acceptance curves  
- Debug visualizations for contours, keypoints, and equalized iris areas  

---

## Conclusions

The developed iris recognition system demonstrates:

- **high accuracy** on CASIA v1.0  
- strong robustness to moderate acquisition variations  
- effective segmentation and matching in most scenarios  

However, performance decreases with:

- strong reflections  
- occlusions (eyelashes, eyelids)  
- low-quality or noisy images  

The system has practical relevance in:

- airport security  
- access control  
- identity verification  
- multimodal biometric systems  

Future improvements may include:

- better eyelash/occlusion handling  
- deep learning–based segmentation  
- optimized keypoint filtering  
- multimodal fusion  

---

## Usage (Visual Studio Code)

### 1. Download CASIA v1.0 dataset  
Download the dataset from the following link:  
https://drive.google.com/drive/u/1/folders/1uPjB4aAZwfOydTtthdO2y4j0HL_O2Fry
**Important:** Extract the contents of `dataset.zip` **directly into the project directory**.  
Do **not** place it inside an extra folder; the extracted files should be in the same directory as `main.py` and `config.py`.

### 2. Install required libraries
```bash
pip install matplotlib numpy opencv-python scikit-learn
```

### 3. (Optional) Adjust parameters in `config.py`

Recommended editable parameters:

- `CONTOURS`
- `EQUALIZATION`
- `KEYPOINTS`
- `MATCHING_THRESHOLD`
- `SHOW_ACCEPTANCES_CURVE`
- `SHOW_REJECTIONS_CURVE`
- `SHOW_MATCHES`
- `SHOW_ROC_CURVE`
- `USE_FIXED_THRESHOLD`
- `USE_HOUGH_LINES_METHOD`

### 4. Run the system

```bash
python main.py
```

**Expected runtime**

- First execution: ~4 hours
- Subsequent executions: ~1 hour (cached `.pkl` and `.txt` files)

**To remove cached files**

Open PowerShell inside the dataset directory and run:

```powershell
Get-ChildItem -Path . -Recurse -Include *.pkl, *.txt | Remove-Item -Force
```

---

**Notes**

- The system uses SIFT, which requires OpenCV’s contrib package if extended functionality is desired.
- Results depend significantly on the segmentation quality.
- The dataset must be correctly placed for the system to function.
