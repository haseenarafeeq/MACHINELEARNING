import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Create a DataFrame from your dataset
data = {
    'Algorithm': ['Naïve Bayes', 'LSTM', 'SVM'],
    'Accuracy': [0.835616438, 0.921875, 0.875]
}
df = pd.DataFrame(data)

# Extract accuracies
nb_acc = df[df['Algorithm']=='Naïve Bayes']['Accuracy']
lstm_acc = df[df['Algorithm']=='LSTM']['Accuracy']  
svm_acc = df[df['Algorithm']=='SVM']['Accuracy']

# T-test between LSTM and Naive Bayes
tstat, pval = ttest_ind(lstm_acc, nb_acc)
print("t-statistic:", tstat, "p-value:", pval)

# Bar plot
algorithms = ['Naive Bayes', 'LSTM', 'SVM']
accuracy = df['Accuracy']

plt.bar(algorithms, accuracy)
plt.title("Algorithm Accuracy")
plt.ylabel("Accuracy")
plt.show()
