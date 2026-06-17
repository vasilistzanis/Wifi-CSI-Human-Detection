# WiFi CSI HAR — Experiment Report

**Generated**: 2026-06-17 17:04:20
**Classes**: no_activity, walk_activity

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| Total recordings | 0 |
| Environments |  (0) |
| Subjects |  (0) |
| Classes | no_activity, walk_activity |

### Recordings per Environment × Class

| Environment | no_activity | walk_activity | Total |
|-------------|-----:|-----:|------:|

## 2. Cross-Environment Evaluation (LEGO)

Leave-Environment-Group-Out: train on one environment, test on the other.
This measures how well the model generalises across physical spaces.

| Experiment | Model | CV (%) | Test Acc (%) | F1 Macro (%) |
|------------|-------|-------:|-------------:|-------------:|
| Train=room1 → Test=livroom | svm | 95.8 ± 2.5 | 87.2 | 86.9 |
| Train=room1 → Test=livroom | rf | 97.0 ± 1.8 | 88.3 | 88.1 |
| Train=room1 → Test=livroom | knn | 92.7 ± 4.0 | 75.7 | 73.6 |
| Train=room1 → Test=livroom | lr | 97.1 ± 1.7 | 87.0 | 86.7 |
| Train=room1 → Test=livroom | et | 96.6 ± 2.5 | 87.8 | 87.7 |
| Train=room1 → Test=livroom | gb | 97.2 ± 1.4 | 88.0 | 87.8 |
| Train=livroom → Test=room1 | svm | 81.3 ± 9.3 | 83.0 | 82.7 |
| Train=livroom → Test=room1 | rf | 80.0 ± 9.7 | 84.0 | 83.7 |
| Train=livroom → Test=room1 | knn | 77.5 ± 7.4 | 85.2 | 85.2 |
| Train=livroom → Test=room1 | lr | 77.2 ± 5.6 | 80.0 | 79.6 |
| Train=livroom → Test=room1 | et | 81.7 ± 8.8 | 90.3 | 90.2 |
| Train=livroom → Test=room1 | gb | 80.2 ± 8.6 | 82.3 | 81.9 |

## 3. Cross-Subject Evaluation (LOSO)

> **Skipped** — requires ≥2 subjects. Currently only 1 subject detected.
> Record data from additional subjects to enable this experiment.

## 4. Ablation Study

Systematic evaluation of individual pipeline components.
Each row disables or varies one parameter from the baseline configuration.

| Configuration | Model | PCA | Window | Diff | Augment | CV (%) | Test Acc (%) | F1 (%) |
|---------------|-------|----:|-------:|:----:|:-------:|-------:|-------------:|-------:|
| Baseline (full pipeline) | svm | 10 | 100 | ✓ | ✓ | 94.0 ± 3.1 | 96.8 | 96.8 |
| Baseline (full pipeline) | rf | 10 | 100 | ✓ | ✓ | 95.7 ± 2.5 | 97.1 | 97.1 |
| Baseline (full pipeline) | knn | 10 | 100 | ✓ | ✓ | 90.1 ± 4.3 | 90.6 | 90.6 |
| Baseline (full pipeline) | lr | 10 | 100 | ✓ | ✓ | 94.6 ± 3.0 | 96.9 | 96.9 |
| Baseline (full pipeline) | et | 10 | 100 | ✓ | ✓ | 95.3 ± 2.9 | 97.1 | 97.1 |
| Baseline (full pipeline) | gb | 10 | 100 | ✓ | ✓ | 95.2 ± 2.8 | 97.2 | 97.2 |
| PCA=5 | svm | 5 | 100 | ✓ | ✓ | 95.2 ± 2.7 | 95.6 | 95.6 |
| PCA=5 | rf | 5 | 100 | ✓ | ✓ | 95.8 ± 2.4 | 96.8 | 96.8 |
| PCA=5 | knn | 5 | 100 | ✓ | ✓ | 92.5 ± 3.3 | 92.8 | 92.8 |
| PCA=5 | lr | 5 | 100 | ✓ | ✓ | 95.7 ± 2.6 | 96.9 | 96.9 |
| PCA=5 | et | 5 | 100 | ✓ | ✓ | 95.7 ± 2.4 | 97.1 | 97.1 |
| PCA=5 | gb | 5 | 100 | ✓ | ✓ | 95.4 ± 2.8 | 96.9 | 96.9 |
| PCA=15 | svm | 15 | 100 | ✓ | ✓ | 94.3 ± 3.1 | 97.0 | 97.0 |
| PCA=15 | rf | 15 | 100 | ✓ | ✓ | 95.7 ± 2.6 | 97.2 | 97.2 |
| PCA=15 | knn | 15 | 100 | ✓ | ✓ | 89.5 ± 4.7 | 90.2 | 90.2 |
| PCA=15 | lr | 15 | 100 | ✓ | ✓ | 94.2 ± 3.2 | 97.3 | 97.3 |
| PCA=15 | et | 15 | 100 | ✓ | ✓ | 95.3 ± 3.0 | 96.9 | 96.9 |
| PCA=15 | gb | 15 | 100 | ✓ | ✓ | 95.4 ± 2.7 | 97.5 | 97.5 |
| PCA=20 | svm | 20 | 100 | ✓ | ✓ | 94.2 ± 3.0 | 96.9 | 96.9 |
| PCA=20 | rf | 20 | 100 | ✓ | ✓ | 95.4 ± 2.9 | 97.2 | 97.2 |
| PCA=20 | knn | 20 | 100 | ✓ | ✓ | 88.4 ± 4.8 | 88.0 | 88.0 |
| PCA=20 | lr | 20 | 100 | ✓ | ✓ | 94.2 ± 3.2 | 96.8 | 96.8 |
| PCA=20 | et | 20 | 100 | ✓ | ✓ | 95.3 ± 2.9 | 96.9 | 96.9 |
| PCA=20 | gb | 20 | 100 | ✓ | ✓ | 95.7 ± 2.6 | 97.5 | 97.5 |
| Window=50 | svm | 10 | 50 | ✓ | ✓ | 93.6 ± 3.3 | 95.8 | 95.8 |
| Window=50 | rf | 10 | 50 | ✓ | ✓ | 95.3 ± 2.7 | 96.4 | 96.4 |
| Window=50 | knn | 10 | 50 | ✓ | ✓ | 88.6 ± 4.8 | 88.5 | 88.5 |
| Window=50 | lr | 10 | 50 | ✓ | ✓ | 94.7 ± 2.9 | 96.6 | 96.6 |
| Window=50 | et | 10 | 50 | ✓ | ✓ | 95.1 ± 2.9 | 96.3 | 96.3 |
| Window=50 | gb | 10 | 50 | ✓ | ✓ | 94.9 ± 2.8 | 96.8 | 96.8 |
| Window=150 | svm | 10 | 150 | ✓ | ✓ | 94.5 ± 2.8 | 96.7 | 96.7 |
| Window=150 | rf | 10 | 150 | ✓ | ✓ | 96.0 ± 2.7 | 97.4 | 97.4 |
| Window=150 | knn | 10 | 150 | ✓ | ✓ | 91.4 ± 4.5 | 92.1 | 92.1 |
| Window=150 | lr | 10 | 150 | ✓ | ✓ | 95.1 ± 2.9 | 96.6 | 96.6 |
| Window=150 | et | 10 | 150 | ✓ | ✓ | 95.6 ± 3.0 | 97.2 | 97.2 |
| Window=150 | gb | 10 | 150 | ✓ | ✓ | 95.7 ± 2.9 | 97.4 | 97.4 |
| Filter=30 | svm | 10 | 100 | ✓ | ✓ | 93.5 ± 2.9 | 95.0 | 95.0 |
| Filter=30 | rf | 10 | 100 | ✓ | ✓ | 94.5 ± 3.5 | 96.1 | 96.1 |
| Filter=30 | knn | 10 | 100 | ✓ | ✓ | 93.0 ± 2.6 | 94.0 | 94.0 |
| Filter=30 | lr | 10 | 100 | ✓ | ✓ | 94.5 ± 2.8 | 95.2 | 95.2 |
| Filter=30 | et | 10 | 100 | ✓ | ✓ | 94.6 ± 3.2 | 96.2 | 96.2 |
| Filter=30 | gb | 10 | 100 | ✓ | ✓ | 94.1 ± 3.8 | 96.5 | 96.5 |
| Filter=50 | svm | 10 | 100 | ✓ | ✓ | 93.6 ± 3.9 | 97.4 | 97.4 |
| Filter=50 | rf | 10 | 100 | ✓ | ✓ | 94.4 ± 3.7 | 96.9 | 96.9 |
| Filter=50 | knn | 10 | 100 | ✓ | ✓ | 92.5 ± 3.4 | 95.9 | 95.9 |
| Filter=50 | lr | 10 | 100 | ✓ | ✓ | 94.0 ± 3.6 | 97.3 | 97.3 |
| Filter=50 | et | 10 | 100 | ✓ | ✓ | 94.5 ± 3.5 | 96.9 | 96.9 |
| Filter=50 | gb | 10 | 100 | ✓ | ✓ | 93.9 ± 4.5 | 97.5 | 97.5 |
| Filter=No Filter | rf | 10 | 100 | ✓ | ✓ | 94.7 ± 3.3 | 96.2 | 96.2 |
| Filter=No Filter | et | 10 | 100 | ✓ | ✓ | 94.7 ± 3.3 | 86.8 | 86.6 |
| Filter=No Filter | knn | 10 | 100 | ✓ | ✓ | 92.9 ± 3.4 | 49.8 | 33.3 |
| Filter=No Filter | lr | 10 | 100 | ✓ | ✓ | 93.6 ± 3.0 | 66.2 | 63.3 |
| Filter=No Filter | svm | 10 | 100 | ✓ | ✓ | 93.5 ± 3.5 | 49.8 | 33.3 |
| Filter=No Filter | gb | 10 | 100 | ✓ | ✓ | 94.0 ± 3.5 | 49.9 | 33.4 |
| Step=25 | svm | 10 | 100 | ✓ | ✓ | 94.0 ± 3.1 | 96.5 | 96.5 |
| Step=25 | rf | 10 | 100 | ✓ | ✓ | 95.5 ± 2.8 | 96.9 | 96.9 |
| Step=25 | knn | 10 | 100 | ✓ | ✓ | 90.1 ± 4.5 | 90.5 | 90.5 |
| Step=25 | lr | 10 | 100 | ✓ | ✓ | 94.9 ± 3.0 | 97.1 | 97.1 |
| Step=25 | et | 10 | 100 | ✓ | ✓ | 95.3 ± 2.9 | 97.0 | 97.0 |
| Step=25 | gb | 10 | 100 | ✓ | ✓ | 95.7 ± 2.8 | 97.4 | 97.4 |
| Step=75 | svm | 10 | 100 | ✓ | ✓ | 94.4 ± 3.1 | 96.0 | 96.0 |
| Step=75 | rf | 10 | 100 | ✓ | ✓ | 94.0 ± 5.3 | 96.8 | 96.8 |
| Step=75 | knn | 10 | 100 | ✓ | ✓ | 89.5 ± 5.1 | 90.2 | 90.1 |
| Step=75 | lr | 10 | 100 | ✓ | ✓ | 95.8 ± 2.3 | 96.0 | 96.0 |
| Step=75 | et | 10 | 100 | ✓ | ✓ | 95.6 ± 2.7 | 96.9 | 96.9 |
| Step=75 | gb | 10 | 100 | ✓ | ✓ | 94.5 ± 4.0 | 96.8 | 96.8 |
| No Temporal Diff | svm | 10 | 100 | ✗ | ✓ | 93.4 ± 4.8 | 94.1 | 94.1 |
| No Temporal Diff | rf | 10 | 100 | ✗ | ✓ | 95.7 ± 3.3 | 97.4 | 97.4 |
| No Temporal Diff | knn | 10 | 100 | ✗ | ✓ | 92.2 ± 4.6 | 92.3 | 92.3 |
| No Temporal Diff | lr | 10 | 100 | ✗ | ✓ | 93.0 ± 5.6 | 95.3 | 95.3 |
| No Temporal Diff | et | 10 | 100 | ✗ | ✓ | 95.6 ± 3.8 | 97.3 | 97.3 |
| No Temporal Diff | gb | 10 | 100 | ✗ | ✓ | 94.3 ± 3.4 | 96.8 | 96.8 |
| No Augmentation | svm | 10 | 100 | ✓ | ✗ | 94.0 ± 3.1 | 96.6 | 96.6 |
| No Augmentation | rf | 10 | 100 | ✓ | ✗ | 95.7 ± 2.5 | 97.2 | 97.2 |
| No Augmentation | knn | 10 | 100 | ✓ | ✗ | 90.1 ± 4.3 | 92.1 | 92.1 |
| No Augmentation | lr | 10 | 100 | ✓ | ✗ | 94.6 ± 3.0 | 96.9 | 96.9 |
| No Augmentation | et | 10 | 100 | ✓ | ✗ | 95.3 ± 2.9 | 97.0 | 97.0 |
| No Augmentation | gb | 10 | 100 | ✓ | ✗ | 95.2 ± 2.8 | 97.2 | 97.2 |
| Augment: noise only | svm | 10 | 100 | ✓ | ✓ | 94.0 ± 3.1 | 96.5 | 96.5 |
| Augment: noise only | rf | 10 | 100 | ✓ | ✓ | 95.7 ± 2.5 | 96.9 | 96.9 |
| Augment: noise only | knn | 10 | 100 | ✓ | ✓ | 90.1 ± 4.3 | 90.0 | 90.0 |
| Augment: noise only | lr | 10 | 100 | ✓ | ✓ | 94.6 ± 3.0 | 96.8 | 96.8 |
| Augment: noise only | et | 10 | 100 | ✓ | ✓ | 95.3 ± 2.9 | 97.1 | 97.1 |
| Augment: noise only | gb | 10 | 100 | ✓ | ✓ | 95.2 ± 2.8 | 97.0 | 97.0 |
| Augment: shift only | svm | 10 | 100 | ✓ | ✓ | 94.0 ± 3.1 | 96.7 | 96.7 |
| Augment: shift only | rf | 10 | 100 | ✓ | ✓ | 95.7 ± 2.5 | 96.9 | 96.9 |
| Augment: shift only | knn | 10 | 100 | ✓ | ✓ | 90.1 ± 4.3 | 90.4 | 90.4 |
| Augment: shift only | lr | 10 | 100 | ✓ | ✓ | 94.6 ± 3.0 | 96.9 | 96.9 |
| Augment: shift only | et | 10 | 100 | ✓ | ✓ | 95.3 ± 2.9 | 97.1 | 97.1 |
| Augment: shift only | gb | 10 | 100 | ✓ | ✓ | 95.2 ± 2.8 | 97.1 | 97.1 |
| Augment: scale only | svm | 10 | 100 | ✓ | ✓ | 94.0 ± 3.1 | 96.7 | 96.7 |
| Augment: scale only | rf | 10 | 100 | ✓ | ✓ | 95.7 ± 2.5 | 97.2 | 97.2 |
| Augment: scale only | knn | 10 | 100 | ✓ | ✓ | 90.1 ± 4.3 | 89.9 | 89.9 |
| Augment: scale only | lr | 10 | 100 | ✓ | ✓ | 94.6 ± 3.0 | 96.9 | 96.9 |
| Augment: scale only | et | 10 | 100 | ✓ | ✓ | 95.3 ± 2.9 | 97.0 | 97.0 |
| Augment: scale only | gb | 10 | 100 | ✓ | ✓ | 95.2 ± 2.8 | 97.1 | 97.1 |
| Augment: time_warp only | svm | 10 | 100 | ✓ | ✓ | 94.0 ± 3.1 | 96.7 | 96.7 |
| Augment: time_warp only | rf | 10 | 100 | ✓ | ✓ | 95.7 ± 2.5 | 96.9 | 96.9 |
| Augment: time_warp only | knn | 10 | 100 | ✓ | ✓ | 90.1 ± 4.3 | 91.2 | 91.2 |
| Augment: time_warp only | lr | 10 | 100 | ✓ | ✓ | 94.6 ± 3.0 | 96.8 | 96.8 |
| Augment: time_warp only | et | 10 | 100 | ✓ | ✓ | 95.3 ± 2.9 | 96.9 | 96.9 |
| Augment: time_warp only | gb | 10 | 100 | ✓ | ✓ | 95.2 ± 2.8 | 97.1 | 97.1 |
| No Diff + No Augment + No Filter | rf | 10 | 100 | ✗ | ✗ | 93.4 ± 4.3 | 92.7 | 92.7 |
| No Diff + No Augment + No Filter | et | 10 | 100 | ✗ | ✗ | 94.6 ± 4.7 | 93.1 | 93.1 |
| No Diff + No Augment + No Filter | knn | 10 | 100 | ✗ | ✗ | 91.0 ± 4.6 | 80.5 | 80.4 |
| No Diff + No Augment + No Filter | lr | 10 | 100 | ✗ | ✗ | 92.4 ± 6.0 | 61.9 | 55.4 |
| No Diff + No Augment + No Filter | svm | 10 | 100 | ✗ | ✗ | 93.1 ± 5.0 | 49.8 | 33.2 |
| No Diff + No Augment + No Filter | gb | 10 | 100 | ✗ | ✗ | 92.8 ± 4.1 | 94.3 | 94.3 |
| Only Time-Domain Features | svm | 10 | 100 | ✓ | ✓ | 94.1 ± 3.0 | 96.5 | 96.5 |
| Only Time-Domain Features | rf | 10 | 100 | ✓ | ✓ | 95.4 ± 2.8 | 96.9 | 96.9 |
| Only Time-Domain Features | knn | 10 | 100 | ✓ | ✓ | 89.6 ± 4.4 | 90.0 | 90.0 |
| Only Time-Domain Features | lr | 10 | 100 | ✓ | ✓ | 94.8 ± 3.0 | 97.1 | 97.1 |
| Only Time-Domain Features | et | 10 | 100 | ✓ | ✓ | 95.2 ± 2.8 | 97.1 | 97.1 |
| Only Time-Domain Features | gb | 10 | 100 | ✓ | ✓ | 95.3 ± 2.6 | 97.0 | 97.0 |
| Only Frequency-Domain Features | svm | 10 | 100 | ✓ | ✓ | 93.7 ± 3.0 | 96.6 | 96.6 |
| Only Frequency-Domain Features | rf | 10 | 100 | ✓ | ✓ | 95.5 ± 2.7 | 96.9 | 96.9 |
| Only Frequency-Domain Features | knn | 10 | 100 | ✓ | ✓ | 91.2 ± 4.4 | 91.2 | 91.2 |
| Only Frequency-Domain Features | lr | 10 | 100 | ✓ | ✓ | 94.9 ± 2.7 | 96.9 | 96.9 |
| Only Frequency-Domain Features | et | 10 | 100 | ✓ | ✓ | 95.2 ± 2.9 | 97.0 | 97.0 |
| Only Frequency-Domain Features | gb | 10 | 100 | ✓ | ✓ | 95.3 ± 3.1 | 97.4 | 97.4 |
| No Zero-Crossing Rate | svm | 10 | 100 | ✓ | ✓ | 93.9 ± 3.0 | 96.5 | 96.5 |
| No Zero-Crossing Rate | rf | 10 | 100 | ✓ | ✓ | 95.5 ± 2.7 | 96.9 | 96.9 |
| No Zero-Crossing Rate | knn | 10 | 100 | ✓ | ✓ | 90.1 ± 4.3 | 90.6 | 90.6 |
| No Zero-Crossing Rate | lr | 10 | 100 | ✓ | ✓ | 94.9 ± 2.9 | 96.9 | 96.9 |
| No Zero-Crossing Rate | et | 10 | 100 | ✓ | ✓ | 95.4 ± 2.9 | 97.1 | 97.1 |
| No Zero-Crossing Rate | gb | 10 | 100 | ✓ | ✓ | 95.3 ± 2.6 | 97.3 | 97.3 |
| No Diff + No Augment | svm | 10 | 100 | ✗ | ✗ | 93.4 ± 4.8 | 94.8 | 94.8 |
| No Diff + No Augment | rf | 10 | 100 | ✗ | ✗ | 95.7 ± 3.3 | 97.2 | 97.2 |
| No Diff + No Augment | knn | 10 | 100 | ✗ | ✗ | 92.2 ± 4.6 | 93.4 | 93.4 |
| No Diff + No Augment | lr | 10 | 100 | ✗ | ✗ | 93.0 ± 5.6 | 95.7 | 95.7 |
| No Diff + No Augment | et | 10 | 100 | ✗ | ✗ | 95.6 ± 3.8 | 97.3 | 97.3 |
| No Diff + No Augment | gb | 10 | 100 | ✗ | ✗ | 94.3 ± 3.4 | 97.3 | 97.3 |
| No Filter + No Augment | svm | 10 | 100 | ✓ | ✗ | 93.5 ± 3.5 | 49.8 | 33.3 |
| No Filter + No Augment | rf | 10 | 100 | ✓ | ✗ | 94.7 ± 3.3 | 94.2 | 94.2 |
| No Filter + No Augment | knn | 10 | 100 | ✓ | ✗ | 92.9 ± 3.4 | 49.9 | 33.6 |
| No Filter + No Augment | lr | 10 | 100 | ✓ | ✗ | 93.6 ± 3.0 | 52.9 | 41.2 |
| No Filter + No Augment | et | 10 | 100 | ✓ | ✗ | 94.7 ± 3.3 | 49.8 | 33.3 |
| No Filter + No Augment | gb | 10 | 100 | ✓ | ✗ | 94.0 ± 3.5 | 49.8 | 33.2 |
| Filter=5.0 | svm | 10 | 100 | ✓ | ✓ | 94.1 ± 3.1 | 96.9 | 96.9 |
| Filter=5.0 | rf | 10 | 100 | ✓ | ✓ | 96.2 ± 2.5 | 97.3 | 97.3 |
| Filter=5.0 | knn | 10 | 100 | ✓ | ✓ | 90.5 ± 4.5 | 90.8 | 90.8 |
| Filter=5.0 | lr | 10 | 100 | ✓ | ✓ | 95.0 ± 3.0 | 97.2 | 97.2 |
| Filter=5.0 | et | 10 | 100 | ✓ | ✓ | 95.2 ± 2.6 | 97.2 | 97.2 |
| Filter=5.0 | gb | 10 | 100 | ✓ | ✓ | 95.5 ± 2.5 | 97.7 | 97.7 |

## 5. Key Findings

- **Cross-Environment**: Best 90.3% (Train=livroom → Test=room1, et), Worst 75.7% (Train=room1 → Test=livroom, knn). Environment gap: **14.6pp**.
- **Ablation Baseline**: 97.2%. Removing components: worst = Filter=No Filter (49.8%), best = Filter=5.0 (97.7%).
-   - *Baseline (full pipeline)*: 90.6% (↓6.6pp vs baseline)
-   - *PCA=5*: 95.6% (↓1.6pp vs baseline)
-   - *PCA=5*: 92.8% (↓4.3pp vs baseline)
-   - *PCA=15*: 90.2% (↓7.0pp vs baseline)
-   - *PCA=20*: 88.0% (↓9.1pp vs baseline)
-   - *Window=50*: 95.8% (↓1.3pp vs baseline)
-   - *Window=50*: 88.5% (↓8.6pp vs baseline)
-   - *Window=150*: 92.1% (↓5.0pp vs baseline)
-   - *Filter=30*: 95.0% (↓2.1pp vs baseline)
-   - *Filter=30*: 96.1% (↓1.0pp vs baseline)
-   - *Filter=30*: 94.0% (↓3.1pp vs baseline)
-   - *Filter=30*: 95.2% (↓2.0pp vs baseline)
-   - *Filter=30*: 96.2% (↓1.0pp vs baseline)
-   - *Filter=50*: 95.9% (↓1.2pp vs baseline)
-   - *Filter=No Filter*: 86.8% (↓10.4pp vs baseline)
-   - *Filter=No Filter*: 49.8% (↓47.4pp vs baseline)
-   - *Filter=No Filter*: 66.2% (↓30.9pp vs baseline)
-   - *Filter=No Filter*: 49.8% (↓47.3pp vs baseline)
-   - *Filter=No Filter*: 49.9% (↓47.3pp vs baseline)
-   - *Step=25*: 90.5% (↓6.7pp vs baseline)
-   - *Step=75*: 96.0% (↓1.2pp vs baseline)
-   - *Step=75*: 90.2% (↓7.0pp vs baseline)
-   - *Step=75*: 96.0% (↓1.1pp vs baseline)
-   - *No Temporal Diff*: 94.1% (↓3.1pp vs baseline)
-   - *No Temporal Diff*: 92.3% (↓4.9pp vs baseline)
-   - *No Temporal Diff*: 95.3% (↓1.8pp vs baseline)
-   - *No Augmentation*: 92.1% (↓5.1pp vs baseline)
-   - *Augment: noise only*: 90.0% (↓7.2pp vs baseline)
-   - *Augment: shift only*: 90.4% (↓6.8pp vs baseline)
-   - *Augment: scale only*: 89.9% (↓7.3pp vs baseline)
-   - *Augment: time_warp only*: 91.2% (↓6.0pp vs baseline)
-   - *No Diff + No Augment + No Filter*: 92.7% (↓4.5pp vs baseline)
-   - *No Diff + No Augment + No Filter*: 93.1% (↓4.1pp vs baseline)
-   - *No Diff + No Augment + No Filter*: 80.5% (↓16.6pp vs baseline)
-   - *No Diff + No Augment + No Filter*: 61.9% (↓35.2pp vs baseline)
-   - *No Diff + No Augment + No Filter*: 49.8% (↓47.4pp vs baseline)
-   - *No Diff + No Augment + No Filter*: 94.3% (↓2.9pp vs baseline)
-   - *Only Time-Domain Features*: 90.0% (↓7.2pp vs baseline)
-   - *Only Frequency-Domain Features*: 91.2% (↓5.9pp vs baseline)
-   - *No Zero-Crossing Rate*: 90.6% (↓6.5pp vs baseline)
-   - *No Diff + No Augment*: 94.8% (↓2.4pp vs baseline)
-   - *No Diff + No Augment*: 93.4% (↓3.8pp vs baseline)
-   - *No Diff + No Augment*: 95.7% (↓1.5pp vs baseline)
-   - *No Filter + No Augment*: 49.8% (↓47.3pp vs baseline)
-   - *No Filter + No Augment*: 94.2% (↓2.9pp vs baseline)
-   - *No Filter + No Augment*: 49.9% (↓47.2pp vs baseline)
-   - *No Filter + No Augment*: 52.9% (↓44.2pp vs baseline)
-   - *No Filter + No Augment*: 49.8% (↓47.3pp vs baseline)
-   - *No Filter + No Augment*: 49.8% (↓47.4pp vs baseline)
-   - *Filter=5.0*: 90.8% (↓6.3pp vs baseline)

---

# 6. Συγκεντρωτικοί Πίνακες Αποτελεσμάτων (Χειροκίνητη Προσθήκη)

## 6.1 Πλήρης Πίνακας Ablation (Accuracy %)

| Διαμόρφωση | SVM (RBF) | RF | ET | GB | LR | k-NN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (πλήρες pipeline) | 96,8 | 97,1 | 97,1 | 97,2 | 96,9 | 90,6 |
| Filter cutoff = 5 Hz | 96,9 | 97,3 | 97,2 | 97,7 | 97,2 | 90,8 |
| Filter cutoff = 30 Hz | 95,0 | 96,1 | 96,2 | 96,5 | 95,2 | 94,0 |
| Filter cutoff = 50 Hz | 97,4 | 96,9 | 96,9 | 97,5 | 97,3 | 95,9 |
| Χωρίς φίλτρο Butterworth | 49,8 | 96,2 | 86,8 | 49,9 | 66,2 | 49,8 |
| PCA = 5 | 95,6 | 96,8 | 97,1 | 96,9 | 96,9 | 92,8 |
| PCA = 15 | 97,0 | 97,2 | 96,9 | 97,5 | 97,3 | 90,2 |
| PCA = 20 | 96,9 | 97,2 | 96,9 | 97,5 | 96,8 | 88,0 |
| Παράθυρο = 50 | 95,8 | 96,4 | 96,3 | 96,8 | 96,6 | 88,5 |
| Παράθυρο = 150 | 96,7 | 97,4 | 97,2 | 97,4 | 96,6 | 92,1 |
| Βήμα = 25 | 96,5 | 96,9 | 97,0 | 97,4 | 97,1 | 90,5 |
| Βήμα = 75 | 96,0 | 96,8 | 96,9 | 96,8 | 96,0 | 90,2 |
| Χωρίς Temporal Diff | 94,1 | 97,4 | 97,3 | 96,8 | 95,3 | 92,3 |
| Χωρίς Augmentation | 96,6 | 97,2 | 97,0 | 97,2 | 96,9 | 92,1 |
| Augmentation: μόνο noise | 96,5 | 96,9 | 97,1 | 97,0 | 96,8 | 90,0 |
| Augmentation: μόνο shift | 96,7 | 96,9 | 97,1 | 97,1 | 96,9 | 90,4 |
| Augmentation: μόνο scale | 96,7 | 97,2 | 97,0 | 97,1 | 96,9 | 89,9 |
| Augmentation: μόνο time_warp | 96,7 | 96,9 | 96,9 | 97,1 | 96,8 | 91,2 |
| Μόνο χαρακτηριστικά χρόνου | 96,5 | 96,9 | 97,1 | 97,0 | 97,1 | 90,0 |
| Μόνο φασματικά χαρακτηριστικά | 96,6 | 96,9 | 97,0 | 97,4 | 96,9 | 91,2 |
| Χωρίς Zero-Crossing Rate | 96,5 | 96,9 | 97,1 | 97,3 | 96,9 | 90,6 |
| Χωρίς Diff + Augment | 94,8 | 97,2 | 97,3 | 97,3 | 95,7 | 93,4 |
| Χωρίς Filter + Augment | 49,8 | 94,2 | 49,8 | 49,8 | 52,9 | 49,9 |
| Χωρίς Diff + Augment + Filter | 49,8 | 92,7 | 93,1 | 94,3 | 61,9 | 80,5 |

## 6.2 Επίδραση Φίλτρου ανά Μοντέλο

| Μοντέλο | Test Acc με φίλτρο (%) | Test Acc χωρίς φίλτρο (%) |
| :--- | :---: | :---: |
| Random Forest | 97,1 | 96,2 |
| Extra Trees | 97,1 | 86,8 |
| Gradient Boosting | 97,2 | 49,9 |
| Logistic Regression | 96,9 | 66,2 |
| SVM (RBF) | 96,8 | 49,8 |
| k-NN | 90,6 | 49,8 |

## 6.3 Αναλυτική Επίδραση στο Random Forest

| Διαμόρφωση (μεταβολή ενός στοιχείου) | Test Acc — RF (%) | Μεταβολή vs baseline |
| :--- | :---: | :---: |
| Baseline (πλήρες pipeline) | 97,1 | — |
| Filter cutoff = 5 Hz | 97,3 | +0,2 |
| Filter cutoff = 30 Hz | 96,1 | −1,0 |
| Filter cutoff = 50 Hz | 96,9 | −0,2 |
| Χωρίς φίλτρο Butterworth | 96,2 | −0,9 |
| PCA = 5 | 96,8 | −0,3 |
| PCA = 15 | 97,2 | +0,1 |
| PCA = 20 | 97,2 | +0,1 |
| Παράθυρο = 50 | 96,4 | −0,7 |
| Παράθυρο = 150 | 97,4 | +0,3 |
| Βήμα = 25 | 96,9 | −0,2 |
| Βήμα = 75 | 96,8 | −0,3 |
| Χωρίς Temporal Diff | 97,4 | +0,3 |
| Χωρίς Augmentation | 97,2 | +0,1 |
| Augmentation: μόνο noise | 96,9 | −0,2 |
| Augmentation: μόνο shift | 96,9 | −0,2 |
| Augmentation: μόνο scale | 97,2 | +0,1 |
| Augmentation: μόνο time_warp | 96,9 | −0,2 |
| Μόνο χαρακτηριστικά χρόνου | 96,9 | −0,2 |
| Μόνο φασματικά χαρακτηριστικά | 96,9 | −0,2 |
| Χωρίς Zero-Crossing Rate | 96,9 | −0,2 |
| Χωρίς Diff + Augment | 97,2 | +0,1 |
| Χωρίς Filter + Augment | 94,2 | −2,9 |
| Χωρίς Diff + Augment + Filter | 92,7 | −4,4 |

## 6.4 Πειράματα Cross-Environment

| Πείραμα | Μοντέλο | CV (%) | Test Acc (%) | F1 Macro (%) |
| :--- | :--- | :---: | :---: | :---: |
| Εκπαίδευση room1 → Έλεγχος livroom | RF | 97,0 ± 1,8 | 88,3 | 88,1 |
| Εκπαίδευση room1 → Έλεγχος livroom | GB | 97,2 ± 1,4 | 88,0 | 87,8 |
| Εκπαίδευση room1 → Έλεγχος livroom | ET | 96,6 ± 2,5 | 87,8 | 87,7 |
| Εκπαίδευση room1 → Έλεγχος livroom | SVM (RBF) | 95,8 ± 2,5 | 87,2 | 86,9 |
| Εκπαίδευση room1 → Έλεγχος livroom | LR | 97,1 ± 1,7 | 87,0 | 86,7 |
| Εκπαίδευση room1 → Έλεγχος livroom | k-NN | 92,7 ± 4,0 | 75,7 | 73,6 |
| Εκπαίδευση livroom → Έλεγχος room1 | ET | 81,7 ± 8,8 | 90,3 | 90,2 |
| Εκπαίδευση livroom → Έλεγχος room1 | k-NN | 77,5 ± 7,4 | 85,2 | 85,2 |
| Εκπαίδευση livroom → Έλεγχος room1 | RF | 80,0 ± 9,7 | 84,0 | 83,7 |
| Εκπαίδευση livroom → Έλεγχος room1 | SVM (RBF) | 81,3 ± 9,3 | 83,0 | 82,7 |
| Εκπαίδευση livroom → Έλεγχος room1 | GB | 80,2 ± 8,6 | 82,3 | 81,9 |
| Εκπαίδευση livroom → Έλεγχος room1 | LR | 77,2 ± 5,6 | 80,0 | 79,6 |
