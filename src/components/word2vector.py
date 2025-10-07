import os, sys

from typing import List, Optional
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from ..exception import CustomException

def tokenize(text : str) -> List[str]:
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def trainW2V(data : pd.DataFrame, savePath : Optional[str] = None) -> None:
    """
    Args:
        dataPath (str): Path to input dataset (expects a column 'sentence')
        savePath (str, optional): Directory or file path to save models.
                                  Defaults to './models' 
    """
    try:
        tokenizedTexts = [tokenize(sentence) for sentence in data['sentence']]
        SGmodel = Word2Vec(
            sentences=tokenizedTexts,
            vector_size=512, 
            window=10, 
            min_count=3, 
            sg=1, 
            workers=-1,
            epochs = 20
        )

        CBOWmodel = Word2Vec(
            sentences=tokenizedTexts,
            vector_size=512,
            window=10,
            min_count=3,
            sg=0,
            workers=-1,
            epochs = 20,
        )

        print(f"✅ Skip-Gram vocab size: {len(SGmodel.wv)}")
        print(f"✅ CBOW vocab size: {len(CBOWmodel.wv)}")

        saveDir = savePath or './models'
        os.makedirs(os.path.dirname(saveDir) if os.path.splitext(saveDir)[1] else saveDir, exist_ok=True)

        if os.path.splitext(saveDir)[1]:
            base = saveDir.replace('.model', '')
            SG_path = f"{base}_sg.model"
            CBOW_path = f"{base}_cbow.model"
        else:
            SG_path = os.path.join(saveDir, 'sg_w2v.model')
            CBOW_path = os.path.join(saveDir, 'cbow_w2v.model')

        SGmodel.save(SG_path)
        CBOWmodel.save(CBOW_path)
        print(f"💾 Models saved at:\n - {SG_path}\n - {CBOW_path}")

    except Exception as e:
        raise CustomException(e, sys)

def getWordEmbeddings(model: Word2Vec, tokens: List[str]) -> np.ndarray:
    """
    Args:
        modelPath (str): Path to saved model.
        tokens (List[str]): List of strings (words) for word embeddings.
    
    Returns:
        np.ndarray : a list of word embeddings.
    """
    try:
        vecs = [model.wv[word] for word in tokens if word in model.wv]
        return np.array(vecs) if vecs else np.zeros((1, model.vector_size))
    except Exception as e:
        raise CustomException(e, sys)