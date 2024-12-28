import re
import string
import math
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

splits = {'train': 'train.csv', 'validation': 'valid.csv', 'test': 'test.csv'}
dftrain = pd.read_csv("hf://datasets/papluca/language-identification/" + splits["train"])
dfval = pd.read_csv("hf://datasets/papluca/language-identification/" + splits["validation"])
dftest = pd.read_csv("hf://datasets/papluca/language-identification/" + splits["test"])

dftrain['prediction']=None
dfval['prediction']=None
dftest['prediction']=None


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



def preprocess_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

def generate_ngrams(text, n):
    return [text[i:i+n] for i in range(len(text)-n+1)]

def create_n_grams(lang_texts, num, k):
    all_ngrams = []
    for text in lang_texts:
        words = preprocess_text(text).replace(' ', '_')
        all_ngrams.extend(generate_ngrams(words, num))
    grams = Counter(all_ngrams)
    sum_freq = sum(grams.values())
    for key in grams.keys():
        red = 1 if '_' not in key else 2
        grams[key] = round(math.log(grams[key] / (red * sum_freq)), 3)
    sorted_grams = sorted(grams.items(), key=lambda x: x[1], reverse=True)
    final_grams = [gram[0] for gram in sorted_grams[:k]]
    log_probs = [gram[1] for gram in sorted_grams[:k]]
    return final_grams, log_probs

def matching_score(test_grams, grams_list):
    dist = {lang: 0 for lang in grams_list.keys()}
    for idx_test, gram in enumerate(test_grams[0]):
        for lang in grams_list.keys():
            if gram in grams_list[lang][0]:
                idx = grams_list[lang][0].index(gram)
                dist[lang] += abs(grams_list[lang][1][idx] - test_grams[1][idx_test])
            else:
                dist[lang] += abs(test_grams[1][idx_test])
    return dist

def detect_language(text, language_profiles, num):
    text_ngrams = create_n_grams([text], num, len(language_profiles))
    bi_dist = matching_score(text_ngrams, language_profiles)
    
    best_match = min(bi_dist, key=bi_dist.get)
    
    return best_match

# Create profiles for each language from your training data
language_profiles = {}
languages = dftrain['labels'].unique()

for lang in languages:
    lang_texts = dftrain[dftrain['labels'] == lang]['text'].tolist()
    language_profiles[lang] = create_n_grams(lang_texts, 3, 100)

# Predict languages for dftrain, dfval, and dftest
for df in [dftrain, dfval, dftest]:
    df['prediction'] = df['text'].apply(lambda x: detect_language(x, language_profiles, 3))

# Example output
print(dftest.head())


eval(dftest)