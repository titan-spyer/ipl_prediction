from abc import ABC, abstractmethod

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Abstract class for missing value Analysis.
#-------------------------------------------
# This is a parent class where other method will inherit from it to perform several actions.
class MissingValue(ABC):
    def analyze(self, df: pd.DataFrame):
        self.identifing_missing_values(df)
        self.visualizing_missing_values(df)
    @abstractmethod
    def identifing_missing_values(self, df: pd.DataFrame):
        pass
    @abstractmethod
    def visualizing_missing_values(self, df: pd.DataFrame):
        pass

# concrete class for missing value detection. 