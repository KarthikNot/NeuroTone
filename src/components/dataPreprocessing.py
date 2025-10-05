import os, sys, re

import pandas as pd
from typing import List
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from ..utils import loadData
from ..exception import CustomException

def cleanText(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def tokenize(text : str) -> List[str]:
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def dataPreprocessing(path : str, savePath : str = None) -> pd.DataFrame:
    ''' 
    Args:
        path (str): refers to the data.csv stored location(path) 
        savePath (str, Optional): refers to the save location(path) of preprocessed data
    '''
    try:
        data = loadData(path)
        data['sentence'] = data['sentence'].apply(cleanText)
        data['tokenizedText'] = data['sentence'].apply(tokenize)
        savePath = './data/processed/processedData.csv' if not savePath else savePath
        if not os.path.exists(savePath):
            os.makedirs(os.path.join('data', 'processed'), exist_ok = True)
        if not savePath.endswith('.csv'): 
            savePath = os.path.join(savePath, 'processedData.csv')
        data.to_csv(savePath, index = False)
        return data
    except Exception as e:
        CustomException(e, sys)