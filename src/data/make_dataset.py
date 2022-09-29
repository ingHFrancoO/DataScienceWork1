# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from preprocessing import process_data
from sklearn.model_selection import train_test_split


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making interim data set from raw data')

    logger.info('reading data')
    data_raw = pd.read_csv(f"{input_filepath}/CellPhoneDS.csv")

    logger.info('pre-processing data')
    processed_data = process_data(data_raw)

    print(f'ready data = {processed_data.shape}')

    logger.info('saving processed data')
    processed_data.reset_index(inplace=True, drop=True)

    X = processed_data.drop("price_range", axis=1)
    y = processed_data["price_range"]

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=123,
        stratify=y
    )

    #x_train.to_parquet(f'{output_filepath}/x_train.parquet', index=False, engine='pyarrow')
    #y_train.to_parquet(f'{output_filepath}/y_train.parquet', index = False)
    #x_test.to_parquet(f'{output_filepath}/x_test.parquet', index = False)
    #y_test.to_parquet(f'{output_filepath}/y_test.parquet', index = False)

    x_train.to_csv(f'{output_filepath}/x_train.csv',
                   index=False, engine='pyarrow')
    y_train.to_csv(f'{output_filepath}/y_train.csv', index=False)
    x_test.to_csv(f'{output_filepath}/x_test.csv', index=False)
    y_test.to_csv(f'{output_filepath}/y_test.csv', index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    main(f'{project_dir}/data/raw', f'{project_dir}/data/interim')
