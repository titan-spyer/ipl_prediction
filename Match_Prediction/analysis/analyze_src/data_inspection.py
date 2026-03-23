from abc import ABC, abstractmethod

import pandas as pd

#Abstract Base class for Data Inspection
#---------------------------------------
#This class is designed to use inheritance in Data Inspection Strategy to find multiple reports
class DataInspection(ABC):
    @abstractmethod
    def inspection(self, df: pd.DataFrame, technique: str):
        """Abstract method for data inspection techniques.

        This method provides a common interface for various data analysis techniques.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to analyze, typically loaded from a CSV file.
        technique : str
            The specific analysis technique to apply.

        Returns
        -------
        None
            This method performs analysis and prints results; it does not return a value.
        """
        pass

#Concrete Class for Numerical Analysis 
#-------------------------------------
#This class provides all the Numerical Analysis functions inside a dataset
class NumericalAnalysis(DataInspection):
    def inspection(self, df: pd.DataFrame, technique: str):
        """Perform numerical analysis on the DataFrame.

        This method analyzes the DataFrame using various numerical inspection techniques.

        Parameters
        ----------
        df : pd.DataFrame
            The pandas DataFrame to perform the action on.
        technique : str
            The analysis technique to apply. Supported values: 'info', 'dtypes', 'missing_values', 'duplicates'.
            If empty or None, defaults to 'info'.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        technique = technique or "info"

        if technique == "info":
            try:
                print("---Printing info of file---")
                print(df.info())
            except Exception as e:
                print(f"Encountered this error: {e}")
        elif technique == "dtypes":
            try:
                print("\n---Printing Data Types---")
                print(df.dtypes)
            except Exception as e:
                print(f"Encountered this error: {e}")
        elif technique == "missing_values":
            try:
                print("\n---Printing Missing Values---")
                print(df.isnull().sum())
            except Exception as e:
                print(f"Encountered this error: {e}")
        elif technique == "duplicates":
            try:
                print("\n---Printing Duplicate Values---")
                print(df.duplicated().sum())
            except Exception as e:
                print(f"Encountered this error: {e}")
        else:
            print(f"Unknown technique: {technique}. Supported: 'info', 'dtypes', 'missing_values', 'duplicates'")

#Concrete class for statistical Analysis
#--------------------------------------
#This class provides all the statistical level analysis.
class StatisticalAnalysis(DataInspection):
    def inspection(self, df: pd.DataFrame, technique: str):
        """Perform statistical analysis on the DataFrame.

        This method analyzes the DataFrame using statistical summary techniques.

        Parameters
        ----------
        df : pd.DataFrame
            The pandas DataFrame to perform the action on.
        technique : str
            The analysis technique to apply. Supported values: 'numerical', 'categorical'.
            If empty or None, defaults to 'numerical'.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        technique = technique or "numerical"

        if technique == "numerical":
            try:
                print("---Printing Numerical Analysis---")
                print(df.describe())
            except Exception as e:
                print(f"Error occurred while performing numerical analysis: {e}.")
        elif technique == "categorical":
            try:
                print("---Printing Categorical Analysis---")
                print(df.describe(include=['O']))
            except Exception as e:
                print(f"Error occurred while performing categorical analysis: {e}.")
        else:
            print(f"Unknown technique: {technique}. Supported: 'numerical', 'categorical'")

#Context Class for Data Inspector.
#---------------------------------
#This class will generate the analysis data.
class DataInspector:
    """Context class for executing data inspection strategies.

    This class uses the Strategy pattern to allow dynamic selection of inspection techniques.
    """
    def __init__(self, strategy: DataInspection):
        """Initialize the DataInspector with a specific inspection strategy.

        Parameters
        ----------
        strategy : DataInspection
            The inspection strategy to use (e.g., NumericalAnalysis or StatisticalAnalysis).
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspection):
        """Set a new inspection strategy.

        Parameters
        ----------
        strategy : DataInspection
            The new inspection strategy to use.
        """
        self._strategy = strategy

    def execute_strategy(self, df: pd.DataFrame, technique: str):
        """Execute the current inspection strategy on the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to analyze.
        technique : str
            The specific technique to apply, as defined by the strategy.
        """
        self._strategy.inspection(df, technique)

if __name__ == "__main__":
    pass
