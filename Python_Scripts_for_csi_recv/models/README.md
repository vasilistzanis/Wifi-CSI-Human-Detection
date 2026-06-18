# WiFi CSI Human Activity Recognition - Models Directory

This directory (`models/`) is the core of the Machine Learning pipeline for the  project. It stores all trained models, data preprocessing transformers (pipelines), experiment results, configurations, and performance visualizations (plots).

---

## 📂 Subdirectories Structure

*   **`experiments/`**
    Contains the detailed results from the Ablation Study and the Cross-Environment/Cross-Subject experiments. 
    *   `experiment_results_FULL.json`: The ultimate source of truth containing all experimental data (confusion matrices, accuracies, etc.) for 24 different configurations.
    *   `experiment_report_FULL.md`: The thesis-ready Markdown report containing the full analysis and summary tables for all 24 experiment variations.

*   **`plots/`**
    Contains all visualizations related to the trained models (e.g., Confusion Matrices, ROC Curves, Feature Importances, comparative Bar Charts, etc.).

*   **`har/` & `motion/`**
    Directories that likely host previous iterations of the models or specialized models separated by problem category (multi-class Human Activity Recognition vs. Binary Motion Detection).

---

## 📄 Directory Files

### 1. Trained Models (`.joblib`)
These files contain the serialized, ready-to-use classification models. They have already been trained on the CSI datasets and are prepared for real-time inference.
*   `rf.joblib`: Random Forest (The most balanced and robust model, Test Acc ~ 97.1%)
*   `et.joblib`: Extra Trees
*   `gb.joblib`: Gradient Boosting
*   `svm.joblib`: Support Vector Machine (RBF Kernel)
*   `knn.joblib`: k-Nearest Neighbors
*   `lr.joblib`: Logistic Regression
*   `mlp.joblib`: Multi-Layer Perceptron (Neural Network)
*   `nb.joblib`: Naive Bayes

### 2. Data Infrastructure (Pipelines)
*   **`csi_pipeline.joblib`**: The most critical file alongside the models. It contains the preprocessing mechanism fitted on the training data (e.g., StandardScaler, PCA). Any new incoming signal from the router **must** pass through this pipeline to be normalized before being fed to the model.
*   **`label_encoder.joblib`**: Stores the mapping of human-readable classes to integers (e.g., `no_activity` → 0, `walk_activity` → 1) and vice-versa, allowing the system to output readable predictions.

### 3. Configurations & Results
*   **`experiment_config.json`**: Records the exact configuration (hyperparameters, paths, random seeds) used during the last central training session of the models above. It ensures 100% reproducibility of the experiments.
*   **`metrics.json`**: Contains a summary of the final performance metrics (Test Accuracy, F1 Score) of the central models.
*   **`.gitignore`**: Ensures that Git handles these large files correctly. Since some models (like `et.joblib` or `knn.joblib`) exceed GitHub's strict file size limit, their version control is managed via **Git LFS (Large File Storage)**.

---

## 🚀 Real-Time Usage (Live Sensing)
When running the live scripts (e.g., `live_dashboard.py` or `live_predict.py`), the following sequence occurs:
1. The `label_encoder.joblib` is loaded to identify the available classes.
2. The `csi_pipeline.joblib` is loaded to transform/compress the live CSI stream into (e.g.) 10-15 PCA components.
3. The desired model (e.g., `rf.joblib`) is loaded into memory.
4. The live stream passes through the pipeline, goes into `rf.predict()`, and the final motion prediction is displayed on the screen!
