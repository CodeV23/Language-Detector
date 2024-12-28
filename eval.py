from dataset import dftest,dftrain,dfval
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def accuracy_function(true_labels, predictions):
    correct_predictions = sum(1 for true, pred in zip(true_labels, predictions) if true == pred)
    accuracy = correct_predictions / len(true_labels) if len(true_labels) > 0 else 0
    return accuracy
def eval(dftest):
    accuracy = accuracy_function(dftest['labels'], dftest['prediction'])
    print(f"Accuracy: {accuracy:.2%}")

    # Count the number of 'unknown' predictions
    unknown_count = (dftest['prediction'] == 'unknown').sum()
    print(f"Number of texts where language couldn't be detected: {unknown_count}")

    # Analyze mismatches
    mismatches = dftest[dftest['labels'] != dftest['prediction']]
    print("\nMismatched predictions:")
    print(mismatches[['labels', 'prediction']].value_counts())

    #confusion matrix
    confusion_matrix = pd.crosstab(dftest['labels'], dftest['prediction'], rownames=['True'], colnames=['Predicted'])

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)

    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    plt.tight_layout()
    plt.show()


