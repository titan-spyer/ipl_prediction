from abc import ABC, abstractmethod

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


class UnivariateAnalysisMethod(ABC):
    """Abstract base for univariate analysis methods."""

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def _validate_column(self, column: str):
        if column not in self.dataframe.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")

    @abstractmethod
    def analyze(self, column: str, strategy: str, **kwargs):
        pass


class AnalysisStrategy(ABC):
    """Abstract strategy interface for plotting one series."""

    @abstractmethod
    def plot(self, series: pd.Series, column: str, **kwargs):
        pass


# --- Numerical strategies ---
class HistogramKDEStrategy(AnalysisStrategy):
    def plot(self, series: pd.Series, column: str, **kwargs):
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        sns.histplot(series.dropna(), kde=True, bins=kwargs.get('bins', 30), color=kwargs.get('color', 'blue'))
        plt.title(f"Histogram + KDE for {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()


class BoxPlotStrategy(AnalysisStrategy):
    def plot(self, series: pd.Series, column: str, **kwargs):
        plt.figure(figsize=kwargs.get('figsize', (8, 5)))
        sns.boxplot(x=series.dropna(), color=kwargs.get('color', 'orange'))
        plt.title(f"Boxplot for {column}")
        plt.xlabel(column)
        plt.show()


# --- Categorical strategies ---
class PieChartStrategy(AnalysisStrategy):
    def plot(self, series: pd.Series, column: str, **kwargs):
        counts = series.dropna().value_counts()
        plt.figure(figsize=kwargs.get('figsize', (7, 7)))
        counts.plot.pie(autopct="%.1f%%", startangle=90, colors=kwargs.get('colors', None))
        plt.ylabel("")
        plt.title(f"Pie chart for {column}")
        plt.axis('equal')
        plt.show()


class DonutChartStrategy(AnalysisStrategy):
    def plot(self, series: pd.Series, column: str, **kwargs):
        counts = series.dropna().value_counts()
        plt.figure(figsize=kwargs.get('figsize', (7, 7)))
        wedges, texts, autotexts = counts.plot.pie(autopct="%.1f%%", startangle=90, wedgeprops={'width': 0.4}, colors=kwargs.get('colors', None), pctdistance=0.75)
        plt.setp(autotexts, size=10, weight="bold")
        plt.ylabel("")
        plt.title(f"Donut chart for {column}")
        plt.axis('equal')
        plt.show()


class VerticalBarChartStrategy(AnalysisStrategy):
    def plot(self, series: pd.Series, column: str, **kwargs):
        counts = series.dropna().value_counts().sort_values(ascending=False)
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        sns.barplot(x=counts.index, y=counts.values, palette=kwargs.get('palette', 'viridis'))
        plt.title(f"Vertical bar chart for {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


class HorizontalBarChartStrategy(AnalysisStrategy):
    def plot(self, series: pd.Series, column: str, **kwargs):
        counts = series.dropna().value_counts().sort_values(ascending=True)
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        sns.barplot(x=counts.values, y=counts.index, palette=kwargs.get('palette', 'viridis'))
        plt.title(f"Horizontal bar chart for {column}")
        plt.xlabel("Count")
        plt.ylabel(column)
        plt.tight_layout()
        plt.show()


class NumericalDataAnalysis(UnivariateAnalysisMethod):
    """Concrete univariate analysis for numerical columns."""

    _strategies = {
        'histogram_kde': HistogramKDEStrategy(),
        'boxplot': BoxPlotStrategy(),
    }

    def analyze(self, column: str, strategy: str = 'histogram_kde', **kwargs):
        self._validate_column(column)
        series = self.dataframe[column]
        if not pd.api.types.is_numeric_dtype(series):
            raise TypeError(f"Column '{column}' is not numeric")

        if strategy not in self._strategies:
            raise ValueError(f"Unsupported numerical strategy '{strategy}'. Supported: {list(self._strategies.keys())}")

        # Descriptive stats
        desc = series.describe()
        print(f"Numerical analysis for '{column}' (strategy={strategy}):")
        print(desc)

        return self._strategies[strategy].plot(series, column, **kwargs)


class CategoricalDataAnalysis(UnivariateAnalysisMethod):
    """Concrete univariate analysis for categorical columns."""

    _strategies = {
        'pie': PieChartStrategy(),
        'donut': DonutChartStrategy(),
        'vertical_bar': VerticalBarChartStrategy(),
        'horizontal_bar': HorizontalBarChartStrategy(),
    }

    def analyze(self, column: str, strategy: str = 'vertical_bar', **kwargs):
        self._validate_column(column)
        series = self.dataframe[column].astype('category')

        if strategy not in self._strategies:
            raise ValueError(f"Unsupported categorical strategy '{strategy}'. Supported: {list(self._strategies.keys())}")

        counts = series.value_counts()
        print(f"Categorical analysis for '{column}' (strategy={strategy}):")
        print(counts)

        return self._strategies[strategy].plot(series, column, **kwargs)


class UnivariateAnalysisContext:
    """Context class to choose data analyzer by type and strategy."""

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self._analyzers = {
            'numerical': NumericalDataAnalysis(dataframe),
            'categorical': CategoricalDataAnalysis(dataframe),
        }

    def analyze(self, column: str, data_type: str, strategy: str, **kwargs):
        data_type = data_type.lower()
        if data_type not in self._analyzers:
            raise ValueError("data_type must be 'numerical' or 'categorical'")

        analyzer = self._analyzers[data_type]
        return analyzer.analyze(column, strategy, **kwargs)

    def available_strategies(self, data_type: str):
        data_type = data_type.lower()
        if data_type not in self._analyzers:
            raise ValueError("data_type must be 'numerical' or 'categorical'")

        return list(self._analyzers[data_type]._strategies.keys())
