
import pandas as pd
import logging
from joblib import load
from sklearn.metrics import accuracy_score
from pathlib import Path
import numpy as np
# libraries to import function from other folder
import sys
# from sys import path 
import os
sys.path.append( os.path.abspath('../../') )
#path.append("../../..")
from src.features.build_features import (print_shape, to_categorical, drop_cols, change_value_colm)

def main(input_filepath, output_filepath, input_test_filepath, report_filepath):
    """ Runs model training scripts to turn processed data from (../processed) into
        a machine learning model (saved in ../models).
    """
    logger = logging.getLogger(__name__)
    logger.info('evaluating ML model')

    model = load(f'{output_filepath}/RF_final_model.joblib')

    x_train = pd.read_parquet(f"{input_filepath}/x_train_model_input.parquet")
    y_train = pd.read_csv(f"{input_filepath}/y_train_model_input.csv")

    y_pred = model.predict(x_train)

    train_score = accuracy_score(y_train, y_pred)
    print(f"Train Score: {train_score}")

    with open(f'{report_filepath}/train_score.txt', 'w') as f:
        f.write(f"Train accuracy Score: {train_score}")

    # test predictions

    x_test = pd.read_parquet(f"{input_test_filepath}/x_test.parquet")
    y_test = pd.read_csv(f"{input_test_filepath}/y_test.csv")

    test = pd.concat([x_test, y_test], axis=1)

    
    test_eval = feature_process(test)
 
    x_test_model = test_eval.drop("price_range", axis=1)
    y_test_model = test_eval["price_range"]

    y_test_pred = model.predict(x_test_model)

    test_score = accuracy_score(y_test_model, y_test_pred)
    print(f"Test Score: {test_score}")

    with open(f'{report_filepath}/test_score.txt', 'w') as f:
        f.write(f"Test accuracy Score: {test_score}")


def feature_process(data: pd.DataFrame):
    """Process raw data into useful files for model."""
    cols_to_drop = [
        'Unnamed: 0',
        'index',
        'werf',
        'hgv',
        'fc'
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
                    )
    return process_data


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main(f'{project_dir}/data/processed',
         f'{project_dir}/models',
         f'{project_dir}/data/interim',
         f'{project_dir}/reports')
