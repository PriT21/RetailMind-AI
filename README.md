# RETAILMIND AI — DAY 1 TO DAY 6 PROJECT GUIDE
**Workspace:** `Incremental Proj/notebooks/train.ipynb`\
**Main tabular dataset:** `data/Telco-Customer-Churn.csv`\
**Image dataset:** `myntradataset/`\
**Purpose:** Learn the complete machine-learning and deep-learning
workflow through one incremental project.

<h2 style="color: #334155; margin: 18px 0 8px 0;">1. Project overview</h2>

> **How to read this journal**
> Each day follows the same structure: **Goal → Why it matters → What to do → Results → Takeaway**.
> The guide distinguishes confirmed results from experiments whose final outputs were not recorded.

RetailMind AI has two practical parts:

1.  **Customer churn prediction:** Use customer details to predict
    whether a telecom customer will leave (`Churn = Yes`) or stay
    (`Churn = No`). This is a binary classification problem.
2.  **Fashion image classification:** Use product images to predict a
    broad fashion category. This is a multi-class image-classification
    problem.

The project develops from basic data preparation and traditional ML
models to neural networks and CNNs.

<h2 style="color: #0F766E; margin: 18px 0 8px 0;">2. Project structure</h2>

``` text
Incremental Proj/
├── data/
│   └── Telco-Customer-Churn.csv
├── myntradataset/
│   ├── images/
│   └── styles.csv
├── models/
│   ├── churn_pipeline.pkl
│   ├── churn_day2_champion.joblib
│   ├── churn_ann_day5.keras
│   └── retail_cnn_day6.keras
├── notebooks/
│   └── train.ipynb
├── src/
│   ├── __init__.py
│   ├── train.py
│   └── predict.py
├── test_prediction.py
├── requirements.txt
├── .gitignore
└── README.md
```

The model filenames above are the artifacts discussed in the project.
Confirm each file exists before treating it as saved. Day 3 and Day 4
final artifacts were not confirmed.

<h2 style="color: #7C3AED; margin: 22px 0 6px 0; padding-bottom: 4px; border-bottom: 1px solid #7C3AED;">DAY 01  |  FOUNDATION — FIRST ML PIPELINE</h2>
 First ML Pipeline: Customer Churn

### Goal

Build a first working classification model that predicts customer churn.
Start with Logistic Regression as a baseline so later models have
something to compare against.

### Dataset Understanding

`Telco-Customer-Churn.csv` contains 7,043 customers and 21 columns. Each
row describes one customer. Columns include tenure, contract, internet
service, monthly charges and total charges. The target is `Churn`, with
`Yes` and `No` values.

### What to Do

1.  **Load the CSV:** Use Pandas `read_csv()`.
2.  **Explore it:** Check `head()`, `shape`, `info()`, `describe()` and
    missing-value counts. Understand rows, columns, data types and
    possible issues.
3.  **Clean the data:** Convert `TotalCharges` to numeric using
    `pd.to_numeric(..., errors="coerce")`. Blank or invalid values
    become missing values.
4.  **Remove the ID:** Drop `customerID` from model inputs. It
    identifies a person but is not a useful churn feature.
5.  **Separate X and y:** `X` contains the customer features. `y`
    contains the churn answer, mapped to 1 for Yes and 0 for No.
6.  **Split the data:** Use an 80/20 train-test split with `stratify=y`.
    The recorded split was 5,634 training rows and 1,409 testing rows.
7.  **Preprocess numerical columns:** Fill missing values with the
    median and scale using `StandardScaler`.
8.  **Preprocess categorical columns:** Fill missing values with the
    most frequent value and use
    `OneHotEncoder(handle_unknown="ignore")`.
9.  **Combine preprocessing:** Use `ColumnTransformer`, then place it
    and Logistic Regression inside a Scikit-learn `Pipeline`.
10. **Train and evaluate:** Fit on training data and evaluate on the
    held-out test data.
11. **Save the complete pipeline:** Save preprocessing and model
    together so future predictions receive the same transformations.

## Why a pipeline?

A pipeline ensures that training and prediction use the same
preprocessing. It also helps prevent data leakage: imputers, scalers and
encoders are fitted on training data, not on the test set.

### Recorded Results

  Metric        Logistic Regression

-- 

  Accuracy                   80.55%
  Precision                  65.72%
  Recall                     55.88%
  ROC-AUC                    84.19%

Saved artifact discussed: `models/churn_pipeline.pkl`.

**Day 1 learning:** Understand the basic supervised-learning workflow:
data inspection → cleaning → feature/target split → train/test split →
preprocessing → model training → evaluation → saving.

<h2 style="color: #0F766E; margin: 22px 0 6px 0; padding-bottom: 4px; border-bottom: 1px solid #0F766E;">DAY 02  |  COMPARISON — CLASSIFICATION MODELS</h2>
 Compare Classification Algorithms

### Goal

Find out whether other algorithms identify churners better than the Day
1 baseline. Do not assume the model with the highest accuracy is
automatically the best.

## Models

-   **Logistic Regression:** The Day 1 baseline; a linear classification
    model.
-   **Decision Tree:** Learns a sequence of feature-based questions and
    decisions.
-   **Random Forest:** Combines predictions from many decision trees.
-   **SVM:** Learns a decision boundary that separates classes.

### What to Do

1.  Reuse the same train/test split and preprocessing approach from
    Day 1. This makes the comparison fair.
2.  Train Decision Tree, Random Forest and SVM alongside the baseline.
3.  Use `class_weight="balanced"` where configured. This gives more
    weight to the less frequent class during learning.
4.  Use `StratifiedKFold` cross-validation. Stratification keeps class
    proportions reasonably similar in each fold.
5.  Compare models using recall as the cross-validation scoring metric.
6.  Evaluate the selected models on the test set using accuracy,
    precision, recall and ROC-AUC.
7.  Select a champion based on the chosen objective and save it
    separately. Keep the Day 1 baseline too.

## Why recall?

Recall answers: "Out of all customers who actually churned, how many did
the model identify?" Missing a customer likely to leave may be costly,
so recall was chosen as the main selection metric. Higher recall can
also mean more false alarms, so the business trade-off matters.

### Cross-Validation Results

  Model             Mean CV Recall

-
  Decision Tree             0.4910
  Random Forest             0.6321
  SVM                       0.7793

### Test-Set Comparison

  Model                   Accuracy   Precision   Recall   ROC-AUC

- 

-- 

-- 

  Logistic Regression       80.55%      65.72%   55.88%    84.19%
  Decision Tree             73.17%      49.46%   48.66%    65.37%
  Random Forest             76.72%      55.42%   62.83%    82.24%
  SVM                       74.38%      51.15%   77.27%    82.11%

**Champion:** SVM, selected because it had the highest cross-validation
recall. Saved artifact discussed: `models/churn_day2_champion.joblib`.

**Day 2 learning:** Different algorithms make different trade-offs.
Select a model using the metric that matters for the problem, not
accuracy alone.

<h2 style="color: #C2410C; margin: 22px 0 6px 0; padding-bottom: 4px; border-bottom: 1px solid #C2410C;">DAY 03  |  OPTIMIZATION — TUNING & GENERALISATION</h2>
 Hyperparameter Tuning, Bias and Variance

### Goal

Improve promising models by searching for better settings, then
understand whether the model is underfitting or overfitting.

## What is a hyperparameter?

A hyperparameter is a setting chosen before model fitting. Examples
include:

-   Decision Tree: `max_depth`, `min_samples_split`, `min_samples_leaf`.
-   Random Forest: `n_estimators`, `max_depth`, `min_samples_leaf`.
-   SVM: `C`, `kernel`, and sometimes `gamma`.

### What to Do

1.  **Create parameter grids:** List a sensible range of settings to try
    for each candidate model.
2.  **Run GridSearchCV:** It tries parameter combinations and evaluates
    them using cross-validation.
3.  **Use recall scoring:** Keep the selection objective consistent with
    Day 2.
4.  **Inspect the search:** Review `best_params_`, `best_score_` and
    `best_estimator_`.
5.  **Compare training and validation scores:** A large gap can indicate
    overfitting.
6.  **Plot learning curves:** Show how training and validation
    performance change as more training examples are used.
7.  **Choose and save the tuned model:** Keep it as a new version rather
    than overwriting earlier artifacts.

## Understand the terms

-   **Underfitting / high bias:** The model performs poorly on both
    training and validation data. It may be too simple.
-   **Overfitting / high variance:** Training performance is very high
    but validation performance is much lower. The model may have
    memorised patterns specific to the training data.
-   **Learning curve:** A plot that helps show whether more data may
    help and how training and validation performance behave.

A filename such as `models/churn_v3.joblib` was discussed as a possible
tuned-model artifact.

**Completion note:** GridSearchCV, bias-variance analysis and learning
curves were discussed, but the final Day 3 scores and saved model were
not confirmed in the project conversation.

**Day 3 learning:** Day 2 asks which algorithm works better. Day 3 asks
how to tune it and whether it generalises to unseen data.

<h2 style="color: #2563EB; margin: 22px 0 6px 0; padding-bottom: 4px; border-bottom: 1px solid #2563EB;">DAY 04  |  FEATURE STRATEGY — ENGINEERING & PCA</h2>
 Feature Engineering, Feature Selection, PCA and Ensembles

### Goal

Improve the information given to the model and experiment with reducing
or transforming the feature set.

## Part A: Feature engineering

Feature engineering means creating useful new columns from existing data
and business knowledge.

Features discussed for churn included:

--
  Feature                             Meaning

-- 

--
  `tenure_group`                      Groups customers by how long they
                                      have stayed

  `avg_monthly_spend`                 Represents average monthly spending

  `is_month_to_month`                 Indicates a month-to-month contract

  `has_fiber`                         Indicates fiber internet service

  `is_electronic_check`               Indicates electronic-check payment

  `has_online_security`               Indicates whether online security
                                      is included

--

**What to do:** Inspect the original columns, create the derived
features, check their values, add them to the model inputs, and compare
results against the original feature set. Only use information that
would be available when making a real prediction.

## Part B: SelectKBest

Feature selection chooses a subset of the original columns.

**What to do:** 1. Choose a suitable scoring function. 2. Choose the
number of features `k`. 3. Fit the selector using training data only. 4.
Transform training and test data using that fitted selector. 5. Train
and evaluate a model. 6. Compare it with the model using all features.

**Why?** Removing less useful inputs can simplify a model and sometimes
improve generalisation.

## Part C: PCA

PCA means Principal Component Analysis. Unlike SelectKBest, PCA does not
simply keep selected original columns. It creates new components that
combine information from the original numerical features.

**What to do:** 1. Prepare and scale numerical features. 2. Fit PCA on
training data only. 3. Choose a number of components or a
variance-retention target. 4. Transform the data. 5. Train a classifier
on the transformed features. 6. Compare performance and the reduced
number of dimensions.

**Why?** PCA can compress information into fewer dimensions. Its
components may be harder to interpret than the original columns.

## Part D: Ensemble learning

An ensemble combines multiple learners. Random Forest is an ensemble of
decision trees. XGBoost is a boosting-based ensemble method.

**What to do:** Train an ensemble, evaluate it using the same split and
metrics, and compare it with earlier models.

**Completion note:** Feature engineering, SelectKBest, PCA, Random
Forest and XGBoost were discussed for Day 4. Final experiment outputs
and a confirmed Day 4 artifact were not available in the project
conversation.

**Day 4 learning:** Feature engineering creates inputs; SelectKBest
selects existing inputs; PCA creates transformed components; ensembles
combine multiple learners.

<h2 style="color: #15803D; margin: 22px 0 6px 0; padding-bottom: 4px; border-bottom: 1px solid #15803D;">DAY 05  |  NEURAL NETWORKS — ANN</h2>
 Artificial Neural Networks (ANN)

### Goal

Understand neural-network basics and apply an ANN to two tasks:
handwritten digit recognition and customer churn prediction.

An ANN contains connected layers of artificial neurons. Each layer
transforms its input and passes it to the next layer.

## Part A: MNIST digit classification

MNIST contains grayscale images of handwritten digits from 0 to 9.

-   Training images: 60,000
-   Testing images: 10,000
-   Image dimensions: 28 × 28
-   Classes: 10

#### What to Do

1.  Load MNIST using TensorFlow/Keras.
2.  Inspect image and label shapes.
3.  Normalise pixel values from 0--255 to 0--1.
4.  Build a network with a `Flatten` layer, Dense hidden layers and a
    10-neuron output layer.
5.  Use ReLU in hidden layers and Softmax in the output layer.
6.  Compile with Adam and sparse categorical cross-entropy.
7.  Train for several epochs and monitor validation performance.
8.  Evaluate on the test set and inspect sample predictions.

### Understand the layers

-   `Flatten`: Converts each 28 × 28 image into a one-dimensional
    vector.
-   `Dense`: Fully connected neural-network layer.
-   `ReLU`: Activation commonly used in hidden layers.
-   `Softmax`: Produces class probabilities for the 10 digits.

#### Recorded Results

  Metric            MNIST ANN

--
  Test Accuracy        97.57%
  Test Loss            0.0840

## Part B: ANN for customer churn

#### What to Do

1.  Return to the Telco dataset.
2.  Keep churn variables separate from MNIST variables. This matters
    because the earlier notebook session reused names such as `X_train`
    and caused data-mixing errors.
3.  Split churn data into training and test sets.
4.  Fit the existing column preprocessing on training data only.
5.  Transform both sets; convert sparse output to dense arrays if
    required.
6.  Build an ANN with an input layer, hidden Dense layers and one output
    neuron.
7.  Use ReLU in hidden layers and Sigmoid in the output layer.
8.  Compile with Adam and binary cross-entropy.
9.  Train with validation data and EarlyStopping.
10. Evaluate using accuracy, precision, recall, F1-score and ROC-AUC.

### MNIST versus churn output

  Item         MNIST                              Churn

- 

--
  Problem      10-class classification            Binary classification
  Output       10 neurons                         1 neuron
  Activation   Softmax                            Sigmoid
  Loss         Sparse categorical cross-entropy   Binary cross-entropy

### Recorded churn ANN result

  Metric           ANN

-- 

--
  Accuracy      79.28%
  Precision     62.20%
  Recall        55.88%
  F1-score      58.87%
  ROC-AUC       83.84%

### Compare SVM and ANN

  Metric        SVM champion      ANN

-- 

-- 

--
  Accuracy            74.38%   79.28%
  Precision           51.15%   62.20%
  Recall              77.27%   55.88%
  F1-score            61.55%   58.87%
  ROC-AUC             82.11%   83.84%

The ANN scored higher in accuracy, precision and ROC-AUC. SVM scored
higher in recall and F1-score. If identifying as many actual churners as
possible is the priority, SVM is stronger on recall. There is no
universally best model; the choice depends on the business objective.

The churn ANN was intended to be saved as `models/churn_ann_day5.keras`;
verify the file exists.

**Day 5 learning:** Understand layers, activations, loss functions,
optimisers, epochs, batches and early stopping. Also learn why output
layers and loss functions change depending on whether a task is binary
or multi-class.

<h2 style="color: #BE185D; margin: 22px 0 6px 0; padding-bottom: 4px; border-bottom: 1px solid #BE185D;">DAY 06  |  COMPUTER VISION — CNN</h2>
 CNN Image Classification

### Goal

Train a Convolutional Neural Network to classify fashion product images
into broad product categories.

A CNN is designed for image data. Convolution filters learn visual
patterns, from simple edges and textures to more complex shapes.

## Part A: Fashion-MNIST exploration

Fashion-MNIST contains 28 × 28 grayscale clothing images, with 60,000
training images and 10,000 test images. It was loaded and inspected
during the Day 6 session.

**Important:** The CNN trained and evaluated in this project session
used the Myntra product images below, not Fashion-MNIST. If a separate
Fashion-MNIST CNN is required by the assignment, that remains a separate
task.

## Part B: Load the Myntra product dataset

Dataset: [Fashion Product Images (Small) 

Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

Extract the dataset into:

``` text
myntradataset/
├── images/
└── styles.csv
```

`styles.csv` contains product metadata such as `id`, `masterCategory`,
`subCategory`, `articleType`, `baseColour`, `season`, `year`, `usage`
and `productDisplayName`.

A CSV parsing error occurred because some rows had inconsistent fields.
It was loaded successfully using the Python parser and skipping
malformed lines:

``` python
products = pd.read_csv(
    DATA_DIR / "styles.csv",
    engine="python",
    on_bad_lines="skip"
)
```

The resulting dataframe had 44,424 rows and 10 columns.

## Part C: Match images and make labels

**What to do:** 1. Build the image path using each product ID. 2. Keep
only records whose image file exists. 3. Use `masterCategory` as the
classification target. 4. Convert category names to integer labels with
`LabelEncoder`. 5. Inspect class counts. 6. Split into training and
testing data using stratification.

Some classes had very few records. A class with only one sample cannot
be split reliably with stratification, so classes with fewer than two
records were filtered out.

### Recorded split

  Item                  Count

--
  Available images     44,418
  Training images      35,534
  Testing images        8,884
  Categories                6

Categories: Accessories, Apparel, Footwear, Free Items, Personal Care
and Sporting Goods.

## Part D: Build the image pipeline

**What to do:** 1. Read each image from its path. 2. Decode it as a
three-channel RGB image. 3. Resize it to 128 × 128 pixels. 4. Scale
pixel values to 0--1. 5. Create TensorFlow datasets for training and
testing. 6. Shuffle the training data. 7. Batch images in groups of 32.
8. Prefetch batches for more efficient input processing.

Key terms: - **Resize:** Makes all images the same dimensions. -
**Normalise:** Scales pixel values to a consistent range. - **Batch:**
Processes a group of images together. - **Shuffle:** Changes the
training order. - **Prefetch:** Prepares later batches while the model
works on the current batch.

## Part E: Data augmentation

The model used: - `RandomFlip("horizontal")` - `RandomRotation(0.1)` -
`RandomZoom(0.1)`

Augmentation creates slightly changed versions of training images. It
helps the model learn patterns that are not dependent on one exact
orientation or framing.

## Part F: Build the CNN

The implemented architecture:

1.  Input: 128 × 128 × 3.
2.  Augmentation layers.
3.  Conv2D with 32 filters.
4.  MaxPooling2D.
5.  Conv2D with 64 filters.
6.  MaxPooling2D.
7.  Conv2D with 128 filters.
8.  MaxPooling2D.
9.  Flatten.
10. Dropout.
11. Dense layer with 128 neurons.
12. Softmax output with 6 neurons.

What each layer does: - **Conv2D:** Learns visual features using
filters. - **MaxPooling2D:** Reduces feature-map dimensions. -
**Flatten:** Converts feature maps to a vector. - **Dropout:** Randomly
disables some units during training to help reduce overfitting. -
**Dense:** Learns combinations of extracted features. - **Softmax:**
Produces probabilities for the six categories.

The model summary showed approximately 3.3 million parameters.

## Part G: Compile and train

The model used: - Optimizer: Adam - Loss: Sparse categorical
cross-entropy - Metric: Accuracy - EarlyStopping: Monitors validation
loss and restores the best weights.

Sparse categorical cross-entropy is appropriate because the labels are
integer class IDs rather than one-hot vectors.

## Part H: Evaluate the CNN

### Recorded test result

  Metric             Score

--
  Test Accuracy     97.81%
  Test Loss         0.0808

### Classification report

  Category           Precision   Recall   F1-score   Support

- 

-- 

-- 

- 

  Accessories             0.96     0.97       0.96     2,255
  Apparel                 0.99     0.99       0.99     4,278
  Footwear                0.99     0.99       0.99     1,844
  Free Items              0.00     0.00       0.00        21
  Personal Care           0.94     0.94       0.94       481
  Sporting Goods          0.60     0.60       0.60         5

### Understand the result

The overall accuracy is high, but the dataset is imbalanced. Apparel has
4,278 test images, while Sporting Goods has only 5 and Free Items has
21. The model performs well on the larger classes but poorly on the rare
classes.

The classification report produced undefined precision warnings because
at least one class had no predicted samples. This is a minority-class
prediction issue, not a failure of the entire model.

The confusion matrix compares actual classes against predicted classes.
Its diagonal entries are correct predictions; off-diagonal entries show
which categories the model confuses.

The CNN was intended to be saved as `models/retail_cnn_day6.keras`;
verify the file exists.

**Day 6 learning:** Understand image loading, resizing, normalisation,
augmentation, CNN layers, model training, classification reports and
confusion matrices.

# 8. Metrics and concepts used throughout the project

--
  Concept                             Simple meaning

-- 

--
  Accuracy                            Fraction of all predictions that
                                      are correct

  Precision                           Of predicted positives, how many
                                      were correct

  Recall                              Of actual positives, how many were
                                      found

  F1-score                            Harmonic balance of precision and
                                      recall

  ROC-AUC                             How well scores separate positive
                                      and negative classes

  Confusion matrix                    Table of actual versus predicted
                                      classes

  Cross-validation                    Repeated train/validation checks
                                      across data folds

  Hyperparameter                      A setting chosen to control model
                                      training

  Data leakage                        Using information from outside the
                                      training process

  Overfitting                         Learning training examples too
                                      specifically

  Underfitting                        Failing to learn important patterns

  Feature engineering                 Creating useful input columns

  Feature selection                   Keeping a subset of existing
                                      columns

  PCA                                 Transforming inputs into fewer
                                      principal components

  Augmentation                        Creating varied training examples
                                      from images

--

## Important rule: prevent data leakage

Fit preprocessing, scaling, feature selection and PCA using training
data only. Then apply the fitted transformation to validation and test
data. Keep the test set for final evaluation rather than repeatedly
tuning against it.

# 9. Day-wise quick checklist

--
  Day                     What you should do      Expected
                                                  learning/output

-- 

-- 

--
  1                       Clean Telco data,       First churn pipeline
                          preprocess, train       
                          Logistic Regression,    
                          evaluate and save       

  2                       Compare Logistic        Champion model selected
                          Regression, Decision    using recall
                          Tree, Random Forest and 
                          SVM using               
                          cross-validation        

  3                       Tune parameters with    Understand model tuning
                          GridSearchCV; inspect   and generalisation
                          bias/variance and       
                          learning curves         

  4                       Create features; try    Understand feature
                          SelectKBest, PCA and    improvement and
                          ensemble models         reduction

  5                       Train ANN on MNIST and  Understand neural
                          churn data; compare ANN networks
                          with SVM                

  6                       Prepare product images, Fashion image
                          augment, train CNN and  classifier
                          inspect classification  
                          report/confusion matrix 

--

# 10. Suggested next improvements

These are future tasks, not completed project results:

1.  Record verified Day 3 tuning scores and save the final tuned model.
2.  Complete Day 4 comparisons and document actual results.
3.  Train a separate Fashion-MNIST CNN if required by the assignment.
4.  Investigate class imbalance in the Myntra dataset and improve
    minority-class evaluation.
5.  Try transfer learning with a pretrained image model.
6.  Expose a saved model through a FastAPI prediction endpoint.
7.  Add tests for prediction and input validation.

<h2 style="color: #15803D; margin: 18px 0 8px 0;">Final project summary</h2>

RetailMind AI moves from a traditional tabular ML pipeline to model
comparison, tuning, feature engineering, ANN and CNN image
classification.

The central lesson is to understand the complete workflow

not just
chase a high accuracy score. Prepare data correctly, avoid leakage,
select meaningful metrics, inspect errors, compare models fairly, and
save reproducible artifacts.

**Main notebook:** `notebooks/train.ipynb`\
**Churn dataset:** `data/Telco-Customer-Churn.csv`\
**Fashion dataset:** `myntradataset/`\
**Model directory:** `models/`

## DOCUMENT NOTES

- This is a learning journal, not a claim that every experiment is production-ready.
- Reported scores are the results recorded during the project work.
- Items marked as unconfirmed should be checked in the notebook or `models/` folder.
- Run notebook cells in order, especially after a kernel restart.

**RetailMind AI**  
*Learn the process. Understand the result. Improve the model.*
