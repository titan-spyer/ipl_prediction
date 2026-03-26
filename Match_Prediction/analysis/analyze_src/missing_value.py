from abc import ABC, abstractmethod

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Abstract class for missing value Analysis.
#-------------------------------------------
# This is a parent class where other methods will inherit from it to perform several actions.
class MissingValue(ABC):
    def analyze(self, df: pd.DataFrame, **kwargs):
        """Full missing-value report with visualizations."""
        self.identifing_missing_values(df, **kwargs)
        self.visualizing_missing_values(df, **kwargs)

    @abstractmethod
    def identifing_missing_values(self, df: pd.DataFrame, **kwargs):
        pass

    @abstractmethod
    def visualizing_missing_values(self, df: pd.DataFrame, **kwargs):
        pass


# Concrete class for missing value detection.
class MissingValueAnalysis(MissingValue):
    def identifing_missing_values(self, df: pd.DataFrame, show_top: int = 20, **kwargs):
        """Print table of missing values and percentage by column."""
        total_missing = df.isnull().sum()
        percent_missing = (total_missing / len(df)) * 100

        missing_df = pd.DataFrame({
            'missing_count': total_missing,
            'missing_percent': percent_missing
        })
        missing_df = missing_df[missing_df['missing_count'] > 0].sort_values(by='missing_percent', ascending=False)

        if missing_df.empty:
            print("No missing values found.")
            return missing_df

        print("Missing value summary:")
        display_df = missing_df.head(show_top)
        print(display_df.to_string())

        if len(missing_df) > show_top:
            print(f"... and {len(missing_df) - show_top} more columns with missing values.")

        return missing_df

    def visualizing_missing_values(self, df: pd.DataFrame, threshold: int = 40, show_bar: bool = True, show_heatmap: bool = True, **kwargs):
        """Visualize missingness using bar and heatmap.

        threshold: number of columns cut to avoid huge heatmaps.
        """
        missing_series = df.isnull().sum()

        if show_bar:
            missing_series = missing_series[missing_series > 0].sort_values(ascending=False)
            if not missing_series.empty:
                plt.figure(figsize=kwargs.get('bar_figsize', (12, 6)))
                sns.barplot(x=missing_series.index, y=missing_series.values, palette=kwargs.get('bar_palette', 'rocket'))
                plt.xticks(rotation=45, ha='right')
                plt.title('Missing Values Count Per Column')
                plt.ylabel('Missing Count')
                plt.xlabel('Columns')
                plt.tight_layout()
                plt.show()
            else:
                print("No columns with missing values to plot in bar chart.")

        if show_heatmap:
            # Use reduced dimensionality for performance
            cols_with_missing = df.columns[df.isnull().any()]
            if len(cols_with_missing) == 0:
                print("No columns with missing values to plot in heatmap.")
            else:
                cols_for_heatmap = cols_with_missing[:threshold]
                subset = df[cols_for_heatmap]
                plt.figure(figsize=kwargs.get('heatmap_figsize', (14, min(16, len(cols_for_heatmap)))))
                sns.heatmap(subset.isnull(), cbar=False, yticklabels=False, cmap='viridis')
                plt.title(f'Missing Value Heatmap (first {len(cols_for_heatmap)} columns)')
                plt.xlabel('Columns')
                plt.tight_layout()
                plt.show()

    def drop_missing(self, df: pd.DataFrame, axis: int = 0, thresh: int = None, subset=None, inplace: bool = False):
        """Drop rows or columns with missing values."""
        if inplace:
            df.dropna(axis=axis, thresh=thresh, subset=subset, inplace=True)
            return df
        return df.dropna(axis=axis, thresh=thresh, subset=subset)

    def fill_missing(self, df: pd.DataFrame, strategy: str = 'median', groupby: str = None, inplace: bool = False):
        """Fill missing values using median/mean/mode or group-specific strategy."""
        result = df.copy() if not inplace else df

        if groupby is not None:
            if groupby not in df.columns:
                raise ValueError(f"Groupby column '{groupby}' not found")
            group = result.groupby(groupby)
            for col in result.columns:
                if result[col].isnull().any() and pd.api.types.is_numeric_dtype(result[col]):
                    if strategy == 'median':
                        result[col] = group[col].transform(lambda x: x.fillna(x.median()))
                    elif strategy == 'mean':
                        result[col] = group[col].transform(lambda x: x.fillna(x.mean()))
                    elif strategy == 'mode':
                        result[col] = group[col].transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x))
                    else:
                        raise ValueError("strategy must be 'median', 'mean', or 'mode' when groupby is provided")
            if inplace:
                return result
            return result

        for col in result.columns:
            if not result[col].isnull().any():
                continue

            if pd.api.types.is_numeric_dtype(result[col]):
                if strategy == 'median':
                    fill_value = result[col].median()
                elif strategy == 'mean':
                    fill_value = result[col].mean()
                elif strategy == 'mode':
                    fill_value = result[col].mode().iloc[0]
                else:
                    raise ValueError("strategy must be 'median', 'mean', or 'mode' for numeric columns")
            else:
                if strategy == 'mode':
                    fill_value = result[col].mode().iloc[0]
                elif strategy == 'constant':
                    fill_value = kwargs.get('constant', 'missing')
                else:
                    # default for categorical is mode
                    fill_value = result[col].mode().iloc[0]

            result[col] = result[col].fillna(fill_value)

        if inplace:
            return result
        return result


# Helper class in same file if you need EDA high-level driver
class MissingValueInspector:
    def __init__(self, strategy: MissingValue = None):
        self.strategy = strategy or MissingValueAnalysis()

    def set_strategy(self, strategy: MissingValue):
        self.strategy = strategy

    def execute_strategy(self, df: pd.DataFrame, **kwargs):
        return self.strategy.analyze(df, **kwargs)
