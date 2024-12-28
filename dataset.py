import pandas as pd

splits = {'train': 'train.csv', 'validation': 'valid.csv', 'test': 'test.csv'}
dftrain = pd.read_csv("hf://datasets/papluca/language-identification/" + splits["train"])
dfval = pd.read_csv("hf://datasets/papluca/language-identification/" + splits["validation"])
dftest = pd.read_csv("hf://datasets/papluca/language-identification/" + splits["test"])

dftrain['prediction']=None
dfval['prediction']=None
dftest['prediction']=None

print(dftest)