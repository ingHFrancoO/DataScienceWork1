import pandas as pd
import numpy as np


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Process raw data into useful files for model."""
    process_data = (data
                    .pipe(print_shape, msg=' Shape original')
                    .pipe(drop_exact_duplicates)
                    .pipe(print_shape, msg=' Shape after remove exact duplicates')
                    .pipe(remplace_atypical_values_with_nan, atypical_values=['-948961565145.0', '??????', 'nan', 'nhbgvfrtd 56gyub', '5285988458456.0'])
                    .pipe(print_shape, msg=' Shape after remplace de atypical values to NaN value')
                    .pipe(drop_row_with_nan)
                    .pipe(print_shape, msg=' Shape after remove row with any NaN value')
                    .pipe(sort_data, col='index')
                    .pipe(drop_duplicates, drop_cols=['index'])
                    .pipe(print_shape, msg=' Shape after remove movil phone model duplicates')
                    )

    return process_data


def print_shape(data: pd.DataFrame, msg: str = 'Shape =') -> pd.DataFrame:
    """Print shape of dataframe."""
    print(f'{data.shape}{msg}')
    return data


def sort_data(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """Sort data by and specific column"""
    data = data.sort_values(by=col, ascending=False)
    return data


def drop_exact_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows from data."""
    return data.drop_duplicates(keep=False)


def drop_duplicates(data: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
    """Drop duplicate rows from data."""
    data = data.drop_duplicates(subset=drop_cols, keep='first')
    return data


def remplace_atypical_values_with_nan(data: pd.DataFrame, atypical_values: list) -> pd.DataFrame:
    """Remplace de atypical values to NaN value."""
    for atypical in atypical_values:
        data.replace(atypical, np.nan, inplace=True)
    return data


def drop_row_with_nan(data: pd.DataFrame) -> pd.DataFrame:
    """delete rows with any NaN value"""
    data.dropna(inplace=True, axis=0)
    return data
