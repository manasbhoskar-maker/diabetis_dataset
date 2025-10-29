# diabetis_dataset
Diabetes Prediction Project
Project Objective
The goal of this project is to build a machine learning model to predict the likelihood of individuals having diabetes based on various health indicators and demographic information.

Data
The dataset used in this project is the diabetes_binary_health_indicators_BRFSS2015.csv file, which contains data from the Behavioral Risk Factor Surveillance System (BRFSS) in 2015. The dataset includes health-related features and a binary target variable indicating whether an individual has diabetes.

Methodology
The project followed a standard machine learning workflow:

Data Loading and Initial Inspection: Loaded the data into a pandas DataFrame and performed basic checks using head(), info(), describe(), and isnull().sum().
Data Cleaning: Handled duplicate rows by removing them from the dataset. Missing values in 'Education' and 'Income' were imputed with the median.
Exploratory Data Analysis (EDA):
Visualized the distribution of the target variable (Diabetes_binary).
Analyzed the relationship between features and the target variable using histograms and box plots.
Explored correlations between features using a heatmap.
Feature Selection: Selected relevant features for the model based on the EDA.
Data Preparation: Split the data into training and testing sets and applied preprocessing steps (scaling numerical features and one-hot encoding categorical features) using StandardScaler and OneHotEncoder within a ColumnTransformer.
Model Selection and Training: Explored several binary classification models, including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (GradientBoostingClassifier, LightGBM, XGBoost). A Logistic Regression model was trained with class_weight='balanced' to address class imbalance.
Model Evaluation: Evaluated the performance of the trained models using metrics such as Accuracy, Precision, Recall, and F1-score.
Key Findings
The dataset contained duplicate rows, which were removed during cleaning.
The target variable (Diabetes_binary) is imbalanced, with significantly more instances of individuals without diabetes.
EDA revealed potential relationships between several features (e.g., HighBP, HighChol, BMI, GenHlth, DiffWalk, Age) and the likelihood of having diabetes.
The initial Logistic Regression model had high accuracy but low recall due to class imbalance.
Training the Logistic Regression model with class_weight='balanced' significantly improved recall at the cost of precision, highlighting the trade-off between these metrics in imbalanced datasets.
Comparison with other models showed varying performance across metrics, with Logistic Regression (with balanced weights) achieving the highest recall among the models explored.
How to Run the Code
Ensure you have the diabetes_binary_health_indicators_BRFSS2015.csv file.
Open the provided Colab notebook.
Run the code cells sequentially. The notebook includes steps for data loading, cleaning, EDA, feature selection, data preparation, model training, and evaluation.
Dependencies
The key libraries used in this project are:

pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
xgboost
These dependencies can be installed using pip if you are not running in a Colab environment.
