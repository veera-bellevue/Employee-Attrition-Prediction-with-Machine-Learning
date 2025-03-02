import pandas as pd
import numpy as np
input_file_path = r"C:\Users\raghu\Downloads\Veera DS\DSC680\Indugu_DSC680_project_2\IBM-HR-Employee-Attrition.csv"
input_df = pd.read_csv(input_file_path)

input_df.head()

#Dropping unneccessary columns
cols_to_drop = ['EmployeeNumber', 'Over18', 'StandardHours']
input_df_after_drop = input_df.drop(columns=cols_to_drop)


import matplotlib.pyplot as plt
import seaborn as sns
# EDA Analysis
#1. Attrition Distribution

attrition_counts = input_df_after_drop['Attrition'].value_counts()
plt.figure(figsize=(8,6))

sns.barplot(x=attrition_counts.index,y=attrition_counts.values)
plt.title('Distribution of Employee Attrition')
plt.xlabel('Attrition')
plt.ylabel('Count')

for i, value in enumerate(attrition_counts.values):
    plt.text(i, value + 10, str(value), ha='center', fontsize=12)
plt.tight_layout()
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(20,16))
numeric_cols = input_df_after_drop.select_dtypes(include=['int64','float64'])
correlation_matrix = numeric_cols.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidth=0.5, fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.show()

#Box plot Age vs Attrition
plt.figure(figsize=(10,6))
sns.boxplot(x='Attrition', y='Age', data=input_df_after_drop)
plt.title('Boxplot of Age vs Attrition', fontsize=16)
plt.xlabel('Attrition', fontsize=14)
plt.ylabel('Age', fontsize=14)
plt.tight_layout()
plt.show()

#Job satisfaction vs Attrition
job_satisfaction_attrition = input_df_after_drop.groupby(['JobSatisfaction', 'Attrition']).size().unstack()
job_satisfaction_attrition.plot(kind='bar', figsize=(10,6), stacked=False, color=['skyblue', 'orange'])
plt.title('Job Satisfaction vs. Attrition', fontsize=16)
plt.xlabel('Job Satisfaction (1=Low, 4 = High)', fontsize=14)
plt.ylabel('Number of Employees', fontsize=14)
plt.legend(title='Attrition', labels=['No', 'Yes'], fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#Department-wise attrition
department_attrition = input_df_after_drop.groupby(['Department', 'Attrition']).size().unstack()
department_attrition.plot(kind='bar', figsize=(10,6), stacked=False, color=['pink', 'green'])
plt.title('Department-wise Attrition', fontsize=16)
plt.xlabel('Department', fontsize=14)
plt.ylabel('Number of Employees', fontsize=14)
plt.legend(title='Attrition',labels=['No', 'Yes'], fontsize=12)
plt.tight_layout()
plt.show()

#Work Experience vs Attrition
plt.figure(figsize=(10,6))
sns.boxplot(x='Attrition',y='TotalWorkingYears', data=input_df_after_drop)
plt.title('Boxplot of Work Experience vs Attrition', fontsize=16)
plt.xlabel('Attrition', fontsize=14)
plt.ylabel('Total Working Years', fontsize=12)
plt.tight_layout()
plt.show()

#Descriptive statisitcs
descriptive_stats=input_df_after_drop.describe(include='all')
print(descriptive_stats)

input_df_after_drop.head()

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Handle missing values
numeric_features = input_df_after_drop.select_dtypes(include=['int64', 'float64']).columns
categorical_features = input_df_after_drop.select_dtypes(include=['object']).columns

# Encode categorical features
encoder = LabelEncoder()

for col in categorical_features:
    input_df_after_drop[col] = encoder.fit_transform(input_df_after_drop[col])

# Separate features and target
y = input_df_after_drop['Attrition'].replace({'True': 1, 'False': 0})
X = input_df_after_drop.drop('Attrition', axis=1)

# print("X",X.isna().sum())
# print("y",y.unique())
# Fill the missing numerical values with median values of their respective columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Fill the missing categorical values with the most frequent values of their respective columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# print(X_train.isna().sum())
# print(X_test.isna().sum())
# print(y_train.isna().sum())
# print(y_test.isna().sum())
# Apply SMOTE for handling class imbalance
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# pip install scikit-learn==1.2.2 imbalanced-learn==0.10.1


# Train Multiple models with hyperparameter tuning
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

models = {
    # 'Logistic Regression': {
    #     'model': LogisticRegression(),
    #     'params': {
    #         'classifier_penalty':['l1','l2'],
    #         'classifier_C':[0.1, 1, 10]
    #     }
    # },
    'Logistic Regression': {
        'estimator': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression())
        ]),
        'param_grid': {
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l2']
        }
    },
    'Decision Tree': {
        'estimator': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier())
        ]),
        'param_grid': {
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_depth': [3, 5, 7, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
    },
    'SVM': {
        'estimator': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', SVC(probability=True, random_state=42))
        ]),
        'param_grid': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['rbf', 'linear'],
            'classifier__gamma': ['scale', 'auto', 0.1]
        }
    },
    'Random Forest': {
        'estimator': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier())
        ]),
        'param_grid': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [5, 10, None],
            'classifier__min_samples_split': [2, 5]
        }
    },
    'Gradient Boosting': {
        'estimator': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier())
        ]),
        'param_grid': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__min_samples_split': [3, 5]
        }
    }
}

best_models = {}
for name, config in models.items():
    print(f"\nTraining {name}...")
    try:
        grid_search = GridSearchCV(
            estimator=config['estimator'],
            param_grid=config['param_grid'],
            cv=5,
            scoring='f1',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"Best Parameters for {name}:")
        print(grid_search.best_params_)
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        print(name)
        if name == 'Logistic Regression':
            # Get the feature names from the preprocessor
            feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

            # Get coefficients
            coefficients = pipeline.named_steps['classifier'].coef_[0]

            # Create visualization
            plt.figure(figsize=(12, 8))
            sorted_idx = np.argsort(np.abs(coefficients))
            pos = np.arange(sorted_idx.shape[0]) + .5

            plt.barh(pos, coefficients[sorted_idx])
            plt.yticks(pos, feature_names[sorted_idx])
            plt.xlabel('Coefficient Value')
            plt.title('Feature Importance (Logistic Regression)')
            plt.tight_layout()
            plt.show()
        elif name == 'Decision Tree':
            # Get feature importances from the classifier step
            # Check if the classifier has feature_importances_ attribute
            classifier = pipeline.named_steps['classifier']
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

                # Sort importances in descending order
                indices = np.argsort(importances)[::-1]

                # Create plot
                plt.figure(figsize=(12, 8))
                plt.title('Feature Importances (Decision Tree)')
                plt.bar(range(len(importances)), importances[indices])
                plt.xticks(range(len(importances)), feature_names[indices], rotation=45, ha='right')
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.tight_layout()
                plt.show()
            else:
                print(f"The classifier '{type(classifier).__name__}' does not support feature importances.")

            # Print feature importances
            for f, imp in zip(feature_names[indices], importances[indices]):
                print(f'{f}: {imp:.4f}')
        elif name == 'Random Forest':
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()

            if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                importances = model.named_steps['classifier'].feature_importances_
                indices = np.argsort(importances)[::-1]

                plt.figure(figsize=(10, 6))
                plt.title("Feature Importances")
                plt.bar(range(len(importances)), importances[indices])
                plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
                plt.tight_layout()
                plt.show()
        elif name == 'SVM':
            print('SVM')
        elif name == 'Gradient Boost':
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()

            if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                importances = model.named_steps['classifier'].feature_importances_
                indices = np.argsort(importances)[::-1]

                plt.figure(figsize=(10, 6))
                plt.title("Feature Importances")
                plt.bar(range(len(importances)), importances[indices])
                plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
                plt.tight_layout()
                plt.show()
        else:
            print('no match')

    except Exception as e:
        print(f"Error training {name}: {str(e)}")



#Evaluate all trained models and compare their performance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
results = {}
# print("y_test",y_test)
# print("y_pred", y_pred)
# print("y_pred_proba", y_pred_proba)
for name, model in best_models.items():
    print(f"\nEvaluating {name}: ")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC':roc_auc_score(y_test, y_pred)
    }
    results[name] = metrics

    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

#Compare models
comparison_df = pd.DataFrame(results).round(4)
print("\nModel Comparison:")
print(comparison_df)

#Plot model comparison
plt.figure(figsize=(12,6))
comparison_df.plot(kind='bar')
plt.title('Model Performance Comparison')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.legend(title='Models', bbox_to_anchor=(1.05,1))
plt.tight_layout()
plt.show()

Logistic_regression_model = {
    'Logistic Regression': {
        'estimator': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression())
        ]),
        'param_grid': {
             'classifier__C':[0.1, 1, 10],
             'classifier__penalty':['l2']
            }
    }
}
#Plot feature importance for Logistic regression
# First fit the pipeline
pipeline.fit(X_train, y_train)

# Get the feature names from the preprocessor
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

# Get coefficients
coefficients = pipeline.named_steps['classifier'].coef_[0]

# Create visualization
plt.figure(figsize=(12, 8))
sorted_idx = np.argsort(np.abs(coefficients))
pos = np.arange(sorted_idx.shape[0]) + .5

plt.barh(pos, coefficients[sorted_idx])
plt.yticks(pos, feature_names[sorted_idx])
plt.xlabel('Coefficient Value')
plt.title('Feature Importance (Logistic Regression)')
plt.tight_layout()
plt.show()


Decision_Tree_model={
    'Decision Tree':{
        'estimator': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier())
        ]),
        'param_grid': {
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_depth': [3,5,7, None],
            'classifier__min_samples_split':[2,5,10],
            'classifier__min_samples_leaf':[1,2,4]
            }
    }
}
# print(classifier)
# Get feature importances
importances = pipeline.named_steps['Decision Tree'].feature_importances_
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

# Sort importances in descending order
indices = np.argsort(importances)[::-1]

# Create plot
plt.figure(figsize=(12, 8))
plt.title('Feature Importances (Decision Tree)')
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), feature_names[indices], rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# Print feature importances
for f, imp in zip(feature_names[indices], importances[indices]):
    print(f'{f}: {imp:.4f}')


svm_model={
    'SVM': {
         'estimator': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', SVC(probability=True, random_state=42))
        ]),
        'param_grid': {
            'classifier__C': [0.1,1,10],
            'classifier__kernel': ['rbf', 'linear'],
            'classifier__gamma':['scale', 'auto', 0.1]
            }
    }
}
Random_Forest_model={
    'Random Forest': {
        'estimator': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier())
        ]),
        'param_grid': {
            'classifier__n_estimators': [100,200],
            'classifier__max_depth': [5,10, None],
            'classifier__min_samples_split':[2,5]
            }
    }
}
Gradient_Boosting_model={
    'Gradient Boosting': {
        'estimator': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier())
        ]),
        'param_grid': {
            'classifier__n_estimators': [100,200],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__min_samples_split':[3,5]
        }
    }
}