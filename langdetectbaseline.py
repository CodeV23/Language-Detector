from langdetect import detect, LangDetectException
from langcodes import *
from eval import *
from dataset import dftrain, dfval, dftest


def normalize_lang_code(code):
    if code in ['zh-cn', 'zh-tw']:
        return 'zh'
    return code

def detect_languages(texts):
    detected = []
    for text in texts:
        try:
            lang = detect(text)
            detected.append(normalize_lang_code(lang))
        except LangDetectException:
            detected.append('unknown')
    return detected

# Apply the function to the DataFrame
dftest['prediction'] = detect_languages(dftest['text'])

# Print the first few rows to verify
print(dftest.head())

eval(dftest)
