
# RetailMind AI

An end-to-end Machine Learning project to predict customer churn.

## Day 1 - Churn Prediction Pipeline

### Dataset
Telco Customer Churn (Kaggle)

### Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

### Workflow
1. Load and inspect dataset.
2. Clean missing values.
3. Split data into training and testing sets.
4. Preprocess numerical and categorical features.
5. Train Logistic Regression model.
6. Evaluate model performance.
7. Save model using Joblib.
8. Create a prediction helper.

### Model
Logistic Regression

### Evaluation Metrics
- Accuracy: Add your result
- Precision: Add your result
- Recall: Add your result
- ROC-AUC: Add your result

### Prediction
The model predicts whether a customer will churn or stay.

Model saved at: `models/churn_pipeline.pkl`


## Day 3 – Hyperparameter Tuning

### Objective
Improve churn prediction through hyperparameter tuning
and bias-variance analysis.

### Techniques
- GridSearchCV
- Stratified K-Fold Cross Validation
- Bias-Variance Analysis
- Learning Curves

### Champion Model
- Algorithm: [Actual winning model]
- CV Recall: [Actual score]Z
- Best Parameters: [Actual parameters]

### Model Artifact
models/churn_v3.joblib

## Day 5 – Neural Networks and ANN

### 1. MNIST Neural Network Demo
- Loaded the MNIST handwritten digit dataset.
- Normalized pixel values between 0 and 1.
- Built an ANN using TensorFlow/Keras.
- Used Flatten, Dense, ReLU and Softmax layers.
- Achieved **97.57% test accuracy**.

### 2. Customer Churn Prediction using ANN
- Reused the Telco Customer Churn dataset.
- Applied numerical scaling and categorical encoding.
- Built an ANN with 64 and 32 hidden neurons.
- Used ReLU activation in hidden layers and Sigmoid in the output layer.
- Used Adam optimizer and Binary Crossentropy loss.
- Applied EarlyStopping to monitor validation loss.

### 3. ANN Evaluation

| Metric | ANN |
|---|---:|
| Accuracy | 79.28% |
| Precision | 62.20% |
| Recall | 55.88% |
| F1 Score | 58.87% |
| ROC-AUC | 83.84% |

### 4. SVM vs ANN Comparison

| Metric | SVM | ANN |
|---|---:|---:|
| Accuracy | 74.38% | 79.28% |
| Precision | 51.15% | 62.20% |
| Recall | 77.27% | 55.88% |
| F1 Score | 61.55% | 58.87% |
| ROC-AUC | 82.11% | 83.84% |

### 5. Conclusion

- ANN achieved better accuracy, precision and ROC-AUC.
- SVM achieved better recall and F1-score.
- SVM is preferable when identifying more potential churn customers is the main priority.
- ANN provides better overall accuracy and precision in this experiment.

### 6. Model Artifact

- `models/churn_ann_day5.keras` – Saved ANN model.