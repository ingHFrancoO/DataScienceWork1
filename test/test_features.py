import pytest
import pandas as pd
import numpy as np

# from src.features.build_features import

# load data


@pytest.fixture
def data_read():
    data_test = pd.read_parquet('data/interim/x_train.parquet')
    return data_test


def test_data(data_read):
    columns_name = [
        'Unnamed: 0', 'index', 'battery_power', 'blue', 'clock_speed',
        'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep',
        'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width',
        'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
        'touch_screen', 'wifi', 'hgv', 'werf'
    ]
    columns = data_read.columns
    assert len(columns) == 24
    assert set(columns_name) == set(columns)
