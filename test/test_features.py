import pytest
import pandas as pd
import numpy as np

from src.data.preprocessing import drop_row_with_nan
from src.features.build_features import change_value_colm

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
    
def test_no_null (data_read):
    data = drop_row_with_nan(data_read)
    
    assert data.isnull().sum().sum() == 0

def test_encoding_columns(data_read):
    cat_cols = [
        'blue',
        'dual_sim',
        'four_g',
        'three_g',
        'touch_screen',
        'wifi',
    ]

    data = change_value_colm(data_read, cat_cols)
    
    assert list(data['blue'].unique()) == ['0', '1']
    assert list(data['dual_sim'].unique()) == ['1', '0']
    assert list(data['four_g'].unique()) == ['0', '1']
    assert list(data['three_g'].unique()) == ['0', '1']
    assert list(data['touch_screen'].unique()) == ['0', '1']
    assert list(data['wifi'].unique()) == ['0', '1']
