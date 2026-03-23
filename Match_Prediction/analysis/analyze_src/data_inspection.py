from abc import ABC, abstractmethod

import pandas as pd

#Abstract Base class for Data Inspection
#---------------------------------------
#This class is design to use inheritance in Data Inspection Strategy to Find multiple reports
class DataInspection(ABC):
    @abstractmethod
    def inspection(self, df: pd.DataFrame, technique: str):
        """
        "This Method is a common method in all Techniques to Analyze the Data"
        Parameters:
        df pd.DataFrame: This the Data take from the csv file
        technique: This is the technique provide which type of analyization you want to performe

        Returns:
        It returns the Panda Data Frame after processing
        """
        pass

#Concrete Class for Numerical Analysis 
#-------------------------------------
#This class provide all the Numerical Analysis function inside a Data set
class NumericalAnalysis(DataInspection):
    def inspection(self, df: pd.DataFrame, technique: str):
        """
        "This Method Analyze the data through it numercal values"

        Parameters:
        df pd.DataFrame: This is a Panda Data Frame to do the Action.
        technique: This will take severl values(ex.)
        """
        #Checking for Data Frames
        if not isinstance(df):
            print("Data Type is not provided")

        #Checking for Techniques Provided or not
        if not technique:
            try:
                print("---Printing info of file---")
                print(df.info())
            except Exception as e:
                print(f"Counterd this error{e}")

        #Print the Info of the DataSet
        if technique == "info":
            try:
                print("---Printing info of file---")
                print(df.info())
            except Exception as e:
                print(f"Counterd this error{e}")
        #Print the Data types
        elif technique == "dtypes":
            try:
                print("\n---Printing Data Types---")
                print(df.dtypes())
            except Exception as e:
                print(f"Counterd this error{e}")
        #Print the missing Values
        elif technique == "missing_values":
            try:
                print("\n---Printing Missing Values---")
                print(df.isnull().sum())
            except Exception as e:
                print(f"Counterd this error{e}")
        #Print the duplicate Values
        elif technique == "duplicates":
            try:
                print("\n---Printing Duplicate Values")
                print(df.duplicated().sum())
            except Exception as e:
                print(f"Counterd this error{e}")

#Concrete class for statstical Analysis
#--------------------------------------
#This class provide all the statistical level Analysis.
class StatisticalAnalysis(DataInspection):
    def inspection(self, df: pd.DataFrame, technique: str):
        """
        "This Method Analyze the data through it numercal values"

        Parameters:
        df pd.DataFrame: This is a Panda Data Frame to do the Action.
        technique: This will take severl values(ex.)
        """
        #Checking for Data Frames
        if not isinstance(df):
            print("Data Type is not provided")

        #Checking for Techniques Provided or not
        if not technique:
            try:
                print("---Printing info of file---")
                print(df.describe())
            except Exception as e:
                print(f"Counterd this error{e}")
        if technique == "Numerical":
            try:
                print("---Printing Numerical Analysis---")
                print(df.describe())
            except Exception as e:
                print(f"Error Occured while Numerical Analysis: {e}.")
        elif technique == "categorical":
            try:
                print("---Printing Categorical Analysis---")
                print(df.describe(include=['O']))
            except Exception as e:
                print(f"Error Occured while Numerical Analysis: {e}.")

#Context Class for Data Inspector.
#---------------------------------
#This class Will generate the Analysis Data.
class DataInspector:
    def __init__(self, strategy: DataInspection):
        self._strategy = strategy
    def set_strategy(self, strategy: DataInspection):
        self._strategy = strategy
    def excute_strategy(self, df: pd.DataFrame):
        self._strategy.inspection(df)

if __name__ == "__main__":
    pass
