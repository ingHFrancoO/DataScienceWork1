# feature encodign for befor of ML
import pandas as pd
import numpy as np
import logging
from pathlib import Path


def main(input_filepath, output_filepath):
    """ 
        Runs data feature engineering scripts to turn interim data from (../interim) into
        cleaned data ready for machine learning (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making interim data set from raw data')

    x = pd.read_parquet(f"{input_filepath}/x_train.parquet")
    y = pd.read_csv(f"{input_filepath}/y_train.csv")

    data = pd.concat([x, y], axis=1)

    """Process raw data into useful files for model."""
    cols_to_drop = [
        'Unnamed: 0',
        'index',
        'werf',
        'hgv'
    ]
    cat_cols = [
        'blue',
        'dual_sim',
        'four_g',
        'three_g',
        'touch_screen',
        'wifi',
    ]

    process_data = (data
                    .pipe(print_shape, msg=' Shape original')
                    .pipe(drop_cols, drop_cols=cols_to_drop)
                    .pipe(print_shape, msg=' Shape after drop cols')
                    .pipe(change_value_colm, colums=cat_cols)
                    .pipe(print_shape, msg=' Shape after change values of categorical cols')
                    .pipe(to_categorical, categorical_cols=cat_cols)

                    # .pipe(encode_categorical)
                    # .pipe(print_shape, msg=' Shape after encode categorical cols')
                    )
    print('')


def print_shape(data: pd.DataFrame, msg: str = 'Shape ='):
    """Print shape of dataframe."""
    print(f'{data.shape}{msg}')
    return data


def drop_cols(data: pd.DataFrame,
              drop_cols: list = None):
    """Drop columns from data."""
    return data.drop(drop_cols, axis=1)


def change_value_colm(data: pd.DataFrame, colums: list):
    for col in colums:
        data[col].replace('1.0', '1', inplace=True)
        data[col].replace('0.0', '0', inplace=True)
    return data


def to_categorical(data: pd.DataFrame, categorical_cols: list):
    """Convert colum to categorical type"""
    for x in categorical_cols:
        data[x] = data[x].astype('category')
    return data


def encode_categorical(data: pd.DataFrame):
    cat_cols = list(data.select_dtypes('object').columns)
    for col in cat_cols:
        data = pd.concat([data.drop(col, axis=1),
                         pd.get_dummies(data[col], prefix=col, drop_first=True)],
                         axis=1)
    return data


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main(f'{project_dir}/data/interim', f'{project_dir}/data/processed')
