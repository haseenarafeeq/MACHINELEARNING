import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load the dataset from the CSV file with a different encoding
df = pd.read_csv('myresult.csv', encoding='ISO-8859-1')  # Try 'latin1' if 'ISO-8859-1' doesn't work

# Calculate the mean F1-Score, Precision, and Recall for each algorithm
algorithm_scores = df.groupby('Algorithm').agg({
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
    df[df['Algorithm'] == 'LSTM']['F1_Score'],
    df[df['Algorithm'] == 'Naïve Bayes']['F1_Score'],
    df[df['Algorithm'] == 'SVM']['F1_Score']
)

f_statistic_precision, p_value_precision = stats.f_oneway(
    df[df['Algorithm'] == 'LSTM']['Precision'],
    df[df['Algorithm'] == 'Naïve Bayes']['Precision'],
    df[df['Algorithm'] == 'SVM']['Precision']
)

f_statistic_recall, p_value_recall = stats.f_oneway(
    df[df['Algorithm'] == 'LSTM']['Recall'],
    df[df['Algorithm'] == 'Naïve Bayes']['Recall'],
    df[df['Algorithm'] == 'SVM']['Recall']
)

# Print F-statistics and p-values for F1-Score
print("\nF-Statistic F1-Score:", f_statistic_f1)
print("p-value F1-Score:", p_value_f1)

# Print F-statistics and p-values for Precision
print("\nF-Statistic Precision:", f_statistic_precision)
print("p-value Precision:", p_value_precision)

# Print F-statistics and p-values for Recall
print("\nF-Statistic Recall:", f_statistic_recall)
print("p-value Recall:", p_value_recall)

# Create a bar plot for ANOVA test results
plt.figure(figsize=(10, 6))
sns.barplot(x=['F1-Score', 'Precision', 'Recall'], y=[f_statistic_f1, f_statistic_precision, f_statistic_recall])
plt.title("ANOVA Test Results")
plt.ylabel("F-Statistic")
plt.ylim(0, max(f_statistic_f1, f_statistic_precision, f_statistic_recall) + 0.1)
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
tukey_f1 = pairwise_tukeyhsd(df['F1_Score'], df['Algorithm'])
tukey_precision = pairwise_tukeyhsd(df['Precision'], df['Algorithm'])
tukey_recall = pairwise_tukeyhsd(df['Recall'], df['Algorithm'])

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
