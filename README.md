### Diabetes Prediction

**Hanshen Yu**

#### Abstract
This project aims to develop diabetes prediction models using machine learning techniques. The dataset used contains the clinical characteristics of the patient and the goal is to make a prediction about whether the patient has diabetes or not. The workflow includes data preprocessing, data visualization, exploratory data analysis (EDA), model training, hyperparameter tuning, and performance evaluation.

#### Rationale
Diabetes is a critical public health problem. Clinical diagnosis currently relies on fasting blood glucose testing (requiring patients to fast for 8 hours), oral glucose tolerance testing (OGTT) (requiring 2-3 hours), and glycated hemoglobin A1c (HbA1c) testing (reflecting the average blood glucose level over the past 2-3 months), all of which have certain limitations. Machine learning prediction has its own advantages. Prediction models based on machine learning can use routine physical examination indicators for early risk assessment, provide immediate prediction results, identify high-risk groups for targeted screening, and reduce medical testing costs, which help healthcare providers make informed decisions and allocate resources effectively.

#### Research Question
Can machine learning models accurately predict diabetes using clinical features such as blood glucose levels, BMI, age, and insulin levels? Which model performs best in this case?

#### Data Sources
This dataset is from the National Institute of Diabetes and Digestive and Kidney Diseases and is about diabetes among Pima Indians. In particular, all patients here were women of Pima Indian ancestry at least 21 years of age

#### Methodology
1.Data preprocessing
Handling invalid null values: Clinical characteristics such as glucose, BMI, and insulin with null values were estimated using medians grouped by target variable (outcome) to address clearly missing null values.
Scaling: The data is preprocessed using StandardScaler to normalize the numerical features.
2.Exploratory data analysis
Distribution analysis: Plots such as count plots, histograms, and KDE plots are used to visualize the feature distribution and its relationship to the target variable. For example, higher blood glucose levels and BMI are strongly associated with diabetes.
Correlation matrix: The heatmap determines the correlation between features such as glucose and BMI and the outcome.
3. Model development
Three supervised learning models are trained and compared:
Logistic regression
Random forest classifier
Support Vector Machine
Each model is encapsulated in a pipeline with preprocessing steps and hyperparameter tuning using GridSearchCV.
4. Evaluation metrics
Classification metrics: accuracy, precision, recall, F1-score, and ROC-AUC are used to evaluate the performance.
Visualization: Confusion matrices and ROC curves are plotted to explain the model behavior.

#### Results
Key Findings:
Feature importance: Glucose and BMI were the most important features in the random forest model, followed by age and insulin.
Model performance:
model | Accuracy | ROC-AUC
--- | --- | ---
Logistic Regression | 0.7208 | 0.8248
Random Forest | 0.8571 | 0.9446
Support Vector Machine | 0.8377 | 0.8974
The random forest classifier outperforms the other models in both accuracy and ROC-AUC metrics.
Visualizations:
Characteristic distribution: Glucose and BMI have a clear distribution in positive and negative cases.
Learning Curve: All models perform stably with more training data and do not suffer from severe underfitting or overfitting

#### Next steps
Ensemble methods: Explore stacking or boosting techniques (e.g. XGBoost, LightGBM) to improve performance.
Data augmentation: Use SMOTE or ADASYN to address imbalance when necessary.

#### Conclusion
The ROC-AUC of random forest classifier on diabetes prediction reaches 0.9446, showing the best performance. This suggests that machine learning can be a useful tool for diabetes screening.
Clinical validation and consideration of additional features (e.g., lifestyle data) is recommended for actual deployment.

### Bibliography 
Data source: National Institute of Diabetes and Digestive and Kidney Diseases.
Use the library: Pandas, OS, NumPy, Matplotlib, Seaborn, Plotly, sklearn, joblib.

##### Contact and Further Information
E-mail: yuhanshen0310@hotmail.com
