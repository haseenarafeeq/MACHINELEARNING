import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from fancyimpute import IterativeImputer

# Load the dataset from the CSV file with a different encoding
df = pd.read_csv('myresult - Copy.csv', encoding='ISO-8859-1')  # Try 'latin1' if 'ISO-8859-1' doesn't work

# Select only the numeric columns for imputation
numeric_columns = df.select_dtypes(include=['number']).columns

# Perform multiple imputation for missing values using IterativeImputer
imputer = IterativeImputer(max_iter=10, random_state=0)
df_imputed = pd.DataFrame(imputer.fit_transform(df[numeric_columns]), columns=numeric_columns)

# Combine imputed data with 'Algorithm' column
df_imputed['Algorithm'] = df['Algorithm']

# Calculate the mean F1-Score, Precision, and Recall for each algorithm
algorithm_scores = df_imputed.groupby('Algorithm').agg({
    'F1_Score': 'mean',
    'Precision': 'mean',
    'Recall': 'mean'
}).reset_index()

# Determine the best algorithm based on F1-Score
best_algorithm_f1 = algorithm_scores.loc[algorithm_scores['F1_Score'].idxmax()]

# Determine the best algorithm based on Precision
best_algorithm_precision = algorithm_scores.loc[algorithm_scores['Precision'].idxmax()]

# Determine the best algorithm based on Recall
best_algorithm_recall = algorithm_scores.loc[algorithm_scores['Recall'].idxmax()]

# Perform ANOVA tests for F1-Score, Precision, and Recall
f_statistic_f1, p_value_f1 = stats.f_oneway(
    df_imputed[df_imputed['Algorithm'] == 'LSTM']['F1_Score'],
    df_imputed[df_imputed['Algorithm'] == 'Naïve Bayes']['F1_Score'],
    df_imputed[df_imputed['Algorithm'] == 'SVM']['F1_Score']
)

f_statistic_precision, p_value_precision = stats.f_oneway(
    df_imputed[df_imputed['Algorithm'] == 'LSTM']['Precision'],
    df_imputed[df_imputed['Algorithm'] == 'Naïve Bayes']['Precision'],
    df_imputed[df_imputed['Algorithm'] == 'SVM']['Precision']
)

f_statistic_recall, p_value_recall = stats.f_oneway(
    df_imputed[df_imputed['Algorithm'] == 'LSTM']['Recall'],
    df_imputed[df_imputed['Algorithm'] == 'Naïve Bayes']['Recall'],
    df_imputed[df_imputed['Algorithm'] == 'SVM']['Recall']
)

# Debug ANOVA calculations (print F-statistics and p-values)
print("F-Statistic F1-Score:", f_statistic_f1)
print("p-value F1-Score:", p_value_f1)
print("F-Statistic Precision:", f_statistic_precision)
print("p-value Precision:", p_value_precision)
print("F-Statistic Recall:", f_statistic_recall)
print("p-value Recall:", p_value_recall)

# Create a bar plot for ANOVA test results
plt.figure(figsize=(10, 6))
sns.barplot(x=['F1-Score', 'Precision', 'Recall'], y=[f_statistic_f1, f_statistic_precision, f_statistic_recall])
plt.title("ANOVA Test Results")
plt.ylabel("F-Statistic")
# Handle NaN or infinite F-statistic values
valid_f_statistic_values = [f for f in [f_statistic_f1, f_statistic_precision, f_statistic_recall] if not pd.isna(f) and not np.isinf(f)]
plt.ylim(0, max(valid_f_statistic_values, default=0) + 0.1)
plt.yscale("log")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=['F1-Score', 'Precision', 'Recall'], y=[p_value_f1, p_value_precision, p_value_recall])
plt.title("ANOVA Test Results")
plt.ylabel("p-value")
plt.yscale("log")
plt.show()

# Print the best algorithms for each metric
print("\nBest Algorithm based on F1-Score:")
print(best_algorithm_f1)
print("\nBest Algorithm based on Precision:")
print(best_algorithm_precision)
print("\nBest Algorithm based on Recall:")
print(best_algorithm_recall)

# Perform Tukey's HSD tests for F1-Score, Precision, and Recall
tukey_f1 = pairwise_tukeyhsd(df_imputed['F1_Score'], df_imputed['Algorithm'])
tukey_precision = pairwise_tukeyhsd(df_imputed['Precision'], df_imputed['Algorithm'])
tukey_recall = pairwise_tukeyhsd(df_imputed['Recall'], df_imputed['Algorithm'])

# Create box plots for Tukey's HSD results
plt.figure(figsize=(10, 6))
tukey_f1.plot_simultaneous()
plt.title("Tukey's HSD Test Results for F1-Score")
plt.show()

plt.figure(figsize=(10, 6))
tukey_precision.plot_simultaneous()
plt.title("Tukey's HSD Test Results for Precision")
plt.show()

plt.figure(figsize=(10, 6))
tukey_recall.plot_simultaneous()
plt.title("Tukey's HSD Test Results for Recall")
plt.show()

# Print Tukey's HSD results for F1-Score
print("\nTukey's HSD Test Results for F1-Score:")
print(tukey_f1)

# Print Tukey's HSD results for Precision
print("\nTukey's HSD Test Results for Precision:")
print(tukey_precision)

# Print Tukey's HSD results for Recall
print("\nTukey's HSD Test Results for Recall:")
print(tukey_recall)
