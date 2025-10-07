import os, sys, re

import pandas as pd
from ..utils import loadData
from ..exception import CustomException

def cleanText(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def dataPreprocessing(path : str, savePath : str = None) -> pd.DataFrame:
    ''' 
    Args:
        path (str): refers to the data.csv stored location(path) 
        savePath (str, Optional): refers to the save location(path) of preprocessed data
    '''
    try:
        data = loadData(path)
        data['sentence'] = data['sentence'].apply(cleanText)
        savePath = './data/processed/processedData.csv' if not savePath else savePath
        if not os.path.exists(savePath):
            os.makedirs(os.path.join('data', 'processed'), exist_ok = True)
        if not savePath.endswith('.csv'): 
            savePath = os.path.join(savePath, 'processedData.csv')
        data.to_csv(savePath, index = False)
        return data
    except Exception as e:
        raise CustomException(e, sys)