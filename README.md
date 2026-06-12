# Adaptive AI Models for Proactive Disease Detection

# Day 1 Progress

## Topics Learned
* Python 3.11 setup
* VS Code
* Jupyter Extension
* Git and GitHub
* Repository management
## Key Learnings
* Setting up a Python development environment
* Using Git for version control
* Creating and managing GitHub repositories
* Organizing project folders for AI projects

# Day 2 Progress

## Topics Learned
* pandas basics
* CSV files
* DataFrames
* Dataset exploration
* shape
* dtypes
* describe()
## Files Added
* Explore.py
* data/students.csv
## Key Learnings
* pandas is used for data analysis
* CSV files store tabular datasets
* DataFrames are table structures
* Understanding dataset shape and data types
* Using describe() for statistical summaries

# Day 3 Progress

## Topics Learned
* Healthcare dataset analysis
* Exploratory Data Analysis (EDA)
* Histograms
* Feature distributions
* Class imbalance
* matplotlib basics
## Files Added
* data/diabetes.csv
* notebooks/eda.ipynb
## Key Learnings
* Understanding real-world medical datasets
* Visualizing feature distributions using histograms
* Identifying class imbalance in healthcare datasets
* Exploring patterns before model building
## Important Observation
The dataset contains:
* 500 non-diabetic patients
* 268 diabetic patients
This creates a class imbalance problem, which is important in machine learning model performance.

# Day 4 Progress

## Topics Learned
* Breast cancer recurrence prediction
* Machine Learning in healthcare
* Research paper analysis
* Evaluation metrics
* Literature review
* Research gap identification
## Files Added
* paper/paper1_summary.md
## Key Learnings
* Machine learning models can assist in predicting breast cancer recurrence.
* Feature selection plays an important role in improving prediction performance.
* Evaluation metrics such as accuracy, precision, recall, and F1-score are essential for assessing model effectiveness.
* Healthcare datasets often contain challenges such as class imbalance and limited data availability.
* Identifying research gaps helps in developing improved disease prediction systems.
## Paper Reviewed
**Machine Learning for Breast Cancer Recurrence Prediction**

# Day 5 Progress

## Topics Learned
* Project presentation
* Research communication
* Documentation
## Files Added
* README.md
## Key Learnings
* Explaining EDA findings effectively
* Presenting research paper summaries
* Demonstrating project progress to mentors
* Maintaining clear project documentation

# Day 6 Progress

## Topics Learned
* Feature Engineering
* BMI Category Binning
* Age Group Labeling
* Glucose-to-Insulin Ratio
* Correlation Analysis
* Seaborn Heatmaps
## Key Learnings
* Feature engineering helps create more informative variables from existing data.
* BMI values can be grouped into health-related categories.
* Age groups help identify trends across different age ranges.
* The glucose-to-insulin ratio captures the relationship between glucose and insulin levels.
* Correlation analysis helps understand relationships between features.
* Heatmaps provide a clear visualization of feature correlations.

# Day 7 Progress

## Topics Learned
* Linear Regression
* Cost Function
* Gradient Descent
* Model Training
* NumPy Implementation
* Cost vs Epochs Visualization
## Files Added
* regression_scratch.py
## Key Learnings
* Linear Regression is used to model relationships between variables.
* NumPy can be used to implement machine learning algorithms from scratch.
* The cost function measures prediction error during training.
* Gradient Descent iteratively updates parameters to minimize cost.
* Epochs represent complete passes through the training data.
* Plotting cost vs epochs helps visualize model convergence.

# Day 8 Progress

## Topics Learned
* Data Preprocessing Pipeline
* Missing Value Imputation
* Median Imputation
* Min-Max Normalization
* Train-Test Split
* Stratified Sampling
* Reproducibility
## Files Added
* pipeline.py
## Key Learnings
* Pipelines automate preprocessing steps.
* Missing values can be handled using median imputation.
* Min-Max normalization scales features between 0 and 1.
* Stratified sampling preserves class distribution.
* Fixed random seeds ensure reproducible experiments.

# Day 9 Progress

## Topics Learned
* Logistic Regression
* Binary Classification
* Model Training
* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
## Files Added
* logistic_regression.py
* results/results_log.csv (added by code in logistic_regression.py)
## Key Learnings
* Logistic Regression is commonly used for binary classification tasks.
* Accuracy measures overall prediction correctness.
* Precision measures the quality of positive predictions.
* Recall measures how many actual positives are identified.
* F1-Score balances precision and recall.
* ROC-AUC evaluates the model's ability to distinguish between classes.

# Day 10 Progress

## Topics Learned
* Exploratory Data Analysis Reporting
* Class Balance Analysis
* Missing Value Handling
* Feature Distribution Analysis
* Insight Extraction
* Git Tagging
## Files Added
* report_eda.md
## Key Learnings
* EDA findings should be documented clearly and concisely.
* Class imbalance can affect model evaluation.
* Median imputation is effective for handling missing values.
* Feature distributions reveal important dataset characteristics.
* Key insights help guide future model development.
* Git tags can be used to mark project milestones.

# Day 11 Progress

## Topics Learned
* Decision Tree Classifier
* GridSearchCV
* Hyperparameter Tuning
* Cross Validation
* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
## Files Added
* decision_tree.py
* results/dt_confusion_matrix.png
## Key Learnings
* Decision Trees classify data using feature-based decision rules.
* GridSearchCV helps find the best hyperparameter values automatically.
* Cross-validation improves model evaluation reliability.
* Confusion matrices help analyze classification errors.
* Accuracy, Precision, Recall, F1-Score, and ROC-AUC are important evaluation metrics.
* Hyperparameter tuning improved model performance.

# Day 12 Progress

## Topics Learned
* Random Forest
* Ensemble Learning
* Feature Importance
* GridSearchCV
* Model Comparison
## Files Added
* random_forest.py
* results/rf_feature_importance.png
## Key Learnings
* Random Forest combines multiple decision trees to improve performance.
* Ensemble learning reduces overfitting compared to a single Decision Tree.
* Feature importance helps identify influential predictors.
* Model comparison helps select the best baseline model.

# Day 13 Progress

## Topics Learned
* Support Vector Machine (SVM)
* RBF Kernel
* Hyperparameter Tuning
* GridSearchCV
* Decision Boundary
* Model Comparison
## Files Added
* svm_model.py
## Key Learnings
* SVM is a powerful classification algorithm for binary classification tasks.
* The RBF kernel enables SVM to learn non-linear decision boundaries.
* Hyperparameters C and gamma significantly affect model performance.
* GridSearchCV helps identify optimal hyperparameter combinations.
* Comparing multiple models helps select the most effective classifier.

# Day 14 Progress

## Topics Learned
* Research Paper Review
* Diabetes Subtype Classification
* Random Forest in Healthcare
* Gap Analysis
* Personalized Medicine
* Early Disease Detection
## Files Added
* gap_analysis.md
## Key Learnings
* Type 2 Diabetes can be divided into multiple clinical subtypes.
* Machine learning can classify patients into disease subgroups using clinical features.
* Random Forest models can achieve high predictive performance in healthcare applications.
* Research papers should be evaluated for limitations and future improvement opportunities.
* Gap analysis helps identify areas where new research can contribute.

# Day 15 Progress

## Topics Learned
* Model Comparison
* Baseline Model Selection
* Performance Evaluation
* Research Validation
* Healthcare AI Model Assessment
## Files Updated
* results/results_log.csv
* paper/comparison.md
## Baseline Results

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.7078   | 0.6047    | 0.4815 | 0.5361   | 0.8067  |
| Decision Tree       | 0.7987   | 0.7447    | 0.6481 | 0.6931   | 0.7921  |
| Random Forest       | 0.7468   | 0.6531    | 0.5926 | 0.6214   | 0.8133  |
| SVM                 | 0.7208   | 0.6341    | 0.4815 | 0.5474   | 0.7970  |

## Mentor Demo Summary
Presented the performance comparison of Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine models on the diabetes prediction dataset.
## Key Findings
* Decision Tree achieved the highest Accuracy, Precision, Recall, and F1-Score.
* Random Forest achieved the highest ROC-AUC score.
* SVM performed moderately but did not outperform Decision Tree.
* Logistic Regression served as the baseline classifier.
## Strongest Baseline Model
**Decision Tree**
Performance:
* Accuracy: 79.87%
* Precision: 74.47%
* Recall: 64.81%
* F1-Score: 69.31%
## Key Learnings
* Multiple evaluation metrics are necessary for healthcare AI systems.
* Accuracy alone is not sufficient for model selection.
* Model comparison helps identify robust baseline models.
* Research validation ensures alignment with project objectives.

