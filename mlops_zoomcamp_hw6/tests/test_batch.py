import pandas as pd
from datetime import datetime
from batch_Q3 import main, read_data, prepare_data

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),           # duration = 9 minutes
        (1, 1, dt(1, 2), dt(1, 10)),                 # duration = 8 minutes
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),        # duration = 0.983333 minutes
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),            # duration = 1441 minutes (should be filtered out)
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    
    categorical = ['PULocationID', 'DOLocationID']
    df_prepared = prepare_data(df, categorical)
    
    expected_data = [
        ('-1', '-1', dt(1, 1), dt(1, 10), 9.0),       # valid
        ('1', '1', dt(1, 2), dt(1, 10), 8.0),         # valid
    ]
    expected_columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration']
    df_expected = pd.DataFrame(expected_data, columns=expected_columns)
    
    pd.testing.assert_frame_equal(df_prepared.reset_index(drop=True), df_expected.reset_index(drop=True), check_dtype=False)

