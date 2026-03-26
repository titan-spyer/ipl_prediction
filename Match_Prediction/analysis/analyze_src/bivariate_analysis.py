from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class BivariatAnalysisMethod(ABC):
    """Abstract base for bivariate analysis methods."""

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def _validate_columns(self, col1: str, col2: str):
        """Validate both columns exist in dataframe."""
        if col1 not in self.dataframe.columns:
            raise ValueError(f"Column '{col1}' not found in dataframe")
        if col2 not in self.dataframe.columns:
            raise ValueError(f"Column '{col2}' not found in dataframe")

    @abstractmethod
    def analyze(self, col1: str, col2: str, strategy: str, **kwargs):
        pass


class BivariateStrategy(ABC):
    """Abstract strategy interface for bivariate plotting."""

    @abstractmethod
    def plot(self, series1: pd.Series, series2: pd.Series, col1: str, col2: str, **kwargs):
        pass


# --- Categorical vs. Categorical Strategies ---
class StackedBarChartStrategy(BivariateStrategy):
    def plot(self, series1: pd.Series, series2: pd.Series, col1: str, col2: str, **kwargs):
        crosstab = pd.crosstab(series1, series2)
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        crosstab.plot(kind='bar', stacked=True, ax=plt.gca(), colormap=kwargs.get('colormap', 'viridis'))
        plt.title(f"Stacked Bar Chart: {col1} vs {col2}")
        plt.xlabel(col1)
        plt.ylabel("Count")
        plt.legend(title=col2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


class CrosstabHeatmapStrategy(BivariateStrategy):
    def plot(self, series1: pd.Series, series2: pd.Series, col1: str, col2: str, **kwargs):
        crosstab = pd.crosstab(series1, series2)
        plt.figure(figsize=kwargs.get('figsize', (10, 8)))
        sns.heatmap(crosstab, annot=True, fmt='d', cmap=kwargs.get('cmap', 'YlOrRd'), cbar_kws={'label': 'Count'})
        plt.title(f"Crosstab Heatmap: {col1} vs {col2}")
        plt.xlabel(col2)
        plt.ylabel(col1)
        plt.tight_layout()
        plt.show()


# --- Categorical vs. Numerical Strategies ---
class BoxPlotStrategy(BivariateStrategy):
    def plot(self, series1: pd.Series, series2: pd.Series, col1: str, col2: str, **kwargs):
        # Determine which is categorical and which is numerical
        data_dict = {col1: series1, col2: series2}
        cat_col = col1 if pd.api.types.is_object_dtype(series1) or pd.api.types.is_categorical_dtype(series1) else col2
        num_col = col2 if cat_col == col1 else col1

        df_temp = pd.DataFrame({cat_col: data_dict[cat_col], num_col: data_dict[num_col]})
        
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        sns.boxplot(data=df_temp, x=cat_col, y=num_col, palette=kwargs.get('palette', 'Set2'))
        plt.title(f"Box Plot: {cat_col} vs {num_col}")
        plt.xlabel(cat_col)
        plt.ylabel(num_col)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


class ViolinPlotStrategy(BivariateStrategy):
    def plot(self, series1: pd.Series, series2: pd.Series, col1: str, col2: str, **kwargs):
        # Determine which is categorical and which is numerical
        data_dict = {col1: series1, col2: series2}
        cat_col = col1 if pd.api.types.is_object_dtype(series1) or pd.api.types.is_categorical_dtype(series1) else col2
        num_col = col2 if cat_col == col1 else col1

        df_temp = pd.DataFrame({cat_col: data_dict[cat_col], num_col: data_dict[num_col]})
        
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        sns.violinplot(data=df_temp, x=cat_col, y=num_col, palette=kwargs.get('palette', 'muted'))
        plt.title(f"Violin Plot: {cat_col} vs {num_col}")
        plt.xlabel(cat_col)
        plt.ylabel(num_col)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


class StripPlotStrategy(BivariateStrategy):
    """Additional: Strip plot for categorical vs numerical."""
    def plot(self, series1: pd.Series, series2: pd.Series, col1: str, col2: str, **kwargs):
        data_dict = {col1: series1, col2: series2}
        cat_col = col1 if pd.api.types.is_object_dtype(series1) or pd.api.types.is_categorical_dtype(series1) else col2
        num_col = col2 if cat_col == col1 else col1

        df_temp = pd.DataFrame({cat_col: data_dict[cat_col], num_col: data_dict[num_col]})
        
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        sns.stripplot(data=df_temp, x=cat_col, y=num_col, palette=kwargs.get('palette', 'husl'), jitter=True, size=6)
        plt.title(f"Strip Plot: {cat_col} vs {num_col}")
        plt.xlabel(cat_col)
        plt.ylabel(num_col)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


# --- Numerical vs. Numerical Strategies ---
class ScatterPlotRegressionStrategy(BivariateStrategy):
    def plot(self, series1: pd.Series, series2: pd.Series, col1: str, col2: str, **kwargs):
        clean_data = pd.DataFrame({col1: series1, col2: series2}).dropna()
        
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        sns.regplot(x=col1, y=col2, data=clean_data, 
                    scatter_kws=kwargs.get('scatter_kws', {'alpha': 0.5, 's': 50}),
                    line_kws=kwargs.get('line_kws', {'color': 'red', 'linewidth': 2}))
        plt.title(f"Scatter Plot with Regression Line: {col1} vs {col2}")
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class CorrelationMatrixHeatmapStrategy(BivariateStrategy):
    def plot(self, series1: pd.Series, series2: pd.Series, col1: str, col2: str, **kwargs):
        df_temp = pd.DataFrame({col1: series1, col2: series2}).dropna()
        corr_matrix = df_temp.corr()
        
        plt.figure(figsize=kwargs.get('figsize', (8, 6)))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap=kwargs.get('cmap', 'coolwarm'), 
                    center=0, vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
        plt.title(f"Correlation Matrix: {col1} vs {col2}")
        plt.tight_layout()
        plt.show()


class JointPlotStrategy(BivariateStrategy):
    """Additional: Joint distribution plot for numerical vs numerical."""
    def plot(self, series1: pd.Series, series2: pd.Series, col1: str, col2: str, **kwargs):
        df_temp = pd.DataFrame({col1: series1, col2: series2}).dropna()
        
        # Using seaborn JointGrid for better visualization
        g = sns.jointplot(data=df_temp, x=col1, y=col2, kind=kwargs.get('kind', 'scatter'), height=8)
        g.set_axis_labels(col1, col2)
        g.fig.suptitle(f"Joint Plot: {col1} vs {col2}", y=1.00)
        plt.tight_layout()
        plt.show()


class CategoricalCategoricalAnalysis(BivariatAnalysisMethod):
    """Concrete bivariate analysis for Categorical vs. Categorical."""

    _strategies = {
        'stacked_bar': StackedBarChartStrategy(),
        'crosstab_heatmap': CrosstabHeatmapStrategy(),
    }

    def analyze(self, col1: str, col2: str, strategy: str = 'stacked_bar', **kwargs):
        self._validate_columns(col1, col2)
        series1 = self.dataframe[col1].astype('category')
        series2 = self.dataframe[col2].astype('category')

        if strategy not in self._strategies:
            raise ValueError(f"Unsupported categorical strategy '{strategy}'. Supported: {list(self._strategies.keys())}")

        # Print summary statistics
        crosstab = pd.crosstab(series1, series2, margins=True)
        print(f"Categorical vs. Categorical: {col1} vs {col2} (strategy={strategy})")
        print(crosstab)
        print()

        return self._strategies[strategy].plot(series1, series2, col1, col2, **kwargs)


class CategoricalNumericalAnalysis(BivariatAnalysisMethod):
    """Concrete bivariate analysis for Categorical vs. Numerical."""

    _strategies = {
        'boxplot': BoxPlotStrategy(),
        'violin': ViolinPlotStrategy(),
        'strip': StripPlotStrategy(),
    }

    def analyze(self, col1: str, col2: str, strategy: str = 'boxplot', **kwargs):
        self._validate_columns(col1, col2)
        
        # Check which column is categorical and which is numerical
        is_cat_1 = pd.api.types.is_object_dtype(self.dataframe[col1]) or pd.api.types.is_categorical_dtype(self.dataframe[col1])
        is_cat_2 = pd.api.types.is_object_dtype(self.dataframe[col2]) or pd.api.types.is_categorical_dtype(self.dataframe[col2])
        
        if not (is_cat_1 != is_cat_2):  # XOR check
            raise TypeError("One column must be categorical and one must be numerical")

        if strategy not in self._strategies:
            raise ValueError(f"Unsupported categorical-numerical strategy '{strategy}'. Supported: {list(self._strategies.keys())}")

        series1 = self.dataframe[col1].dropna()
        series2 = self.dataframe[col2].dropna()

        # Align both series
        mask = series1.index.isin(series2.index) & series2.index.isin(series1.index)
        series1 = series1[mask]
        series2 = series2[mask]

        # Print summary statistics
        cat_col = col1 if is_cat_1 else col2
        num_col = col2 if is_cat_1 else col1
        print(f"Categorical vs. Numerical: {cat_col} vs {num_col} (strategy={strategy})")
        print(f"Grouped statistics:\n{pd.DataFrame({cat_col: series1 if is_cat_1 else series2, num_col: series2 if is_cat_1 else series1}).groupby(cat_col)[num_col].describe()}")
        print()

        return self._strategies[strategy].plot(series1, series2, col1, col2, **kwargs)


class NumericalNumericalAnalysis(BivariatAnalysisMethod):
    """Concrete bivariate analysis for Numerical vs. Numerical."""

    _strategies = {
        'scatter_regression': ScatterPlotRegressionStrategy(),
        'correlation_heatmap': CorrelationMatrixHeatmapStrategy(),
        'joint_plot': JointPlotStrategy(),
    }

    def analyze(self, col1: str, col2: str, strategy: str = 'scatter_regression', **kwargs):
        self._validate_columns(col1, col2)
        
        series1 = self.dataframe[col1]
        series2 = self.dataframe[col2]
        
        if not (pd.api.types.is_numeric_dtype(series1) and pd.api.types.is_numeric_dtype(series2)):
            raise TypeError(f"Both columns must be numeric. Got {col1}: {series1.dtype}, {col2}: {series2.dtype}")

        if strategy not in self._strategies:
            raise ValueError(f"Unsupported numerical strategy '{strategy}'. Supported: {list(self._strategies.keys())}")

        # Print summary statistics
        clean_data = pd.DataFrame({col1: series1, col2: series2}).dropna()
        correlation = clean_data[col1].corr(clean_data[col2])
        
        print(f"Numerical vs. Numerical: {col1} vs {col2} (strategy={strategy})")
        print(f"Pearson Correlation: {correlation:.4f}")
        print(f"Sample size: {len(clean_data)}")
        print()

        return self._strategies[strategy].plot(series1, series2, col1, col2, **kwargs)


class BivariatAnalysisContext:
    """Context class to choose bivariate analyzer by data types and strategy."""

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self._analyzers = {
            'categorical_categorical': CategoricalCategoricalAnalysis(dataframe),
            'categorical_numerical': CategoricalNumericalAnalysis(dataframe),
            'numerical_numerical': NumericalNumericalAnalysis(dataframe),
        }

    def _infer_data_type(self, col: str) -> str:
        """Infer if column is numerical or categorical."""
        if pd.api.types.is_numeric_dtype(self.dataframe[col]):
            return 'numerical'
        else:
            return 'categorical'

    def analyze(self, col1: str, col2: str, strategy: str = None, analysis_type: str = None, **kwargs):
        """
        Analyze relationship between two columns.
        
        Args:
            col1: First column name
            col2: Second column name
            strategy: Visualization strategy (auto-determined if None)
            analysis_type: 'categorical_categorical', 'categorical_numerical', 'numerical_numerical' (auto-inferred if None)
            **kwargs: Additional plotting parameters
        """
        # Auto-infer analysis type if not provided
        if analysis_type is None:
            type1 = self._infer_data_type(col1)
            type2 = self._infer_data_type(col2)
            
            if type1 == 'categorical' and type2 == 'categorical':
                analysis_type = 'categorical_categorical'
            elif (type1 == 'categorical' and type2 == 'numerical') or (type1 == 'numerical' and type2 == 'categorical'):
                analysis_type = 'categorical_numerical'
            else:
                analysis_type = 'numerical_numerical'
        
        if analysis_type not in self._analyzers:
            raise ValueError(f"analysis_type must be one of {list(self._analyzers.keys())}")
        
        analyzer = self._analyzers[analysis_type]
        
        # Use default strategy if not provided
        if strategy is None:
            strategy = list(analyzer._strategies.keys())[0]
        
        return analyzer.analyze(col1, col2, strategy, **kwargs)

    def available_strategies(self, analysis_type: str):
        """List available strategies for a given analysis type."""
        if analysis_type not in self._analyzers:
            raise ValueError(f"analysis_type must be one of {list(self._analyzers.keys())}")
        
        return list(self._analyzers[analysis_type]._strategies.keys())

    def available_analysis_types(self):
        """List all available analysis types."""
        return list(self._analyzers.keys())
