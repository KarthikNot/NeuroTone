import os, sys
import pandas as pd
from typing import Optional
from gensim.models import Word2Vec
from .exception import CustomException

def loadData(filePath: str) -> Optional[pd.DataFrame]:
    """
    Args:
        filePath (str): Path to the CSV file.

    Returns:
        Optional[pd.DataFrame]: Loaded DataFrame if the file exists, else None.
    """
    try:
        if os.path.exists(filePath):
            return pd.read_csv(filePath)
        print(f"❌ File not found at {filePath}")
        return None
    except Exception as e:
        raise CustomException(e, sys)