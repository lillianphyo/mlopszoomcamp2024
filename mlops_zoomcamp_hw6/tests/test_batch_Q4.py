import os
import pandas as pd
from datetime import datetime
# from homework.batch import prepare_data, read_data
from batch_Q4 import main, read_data, prepare_data

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



def test_read_data_from_s3():
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
    bucket_name = 'nyc-duration'
    file_name = 'in/2023-03.parquet'
    
    # Set up the file in the mock S3
    # ... (Assume the file is correctly uploaded to the mock S3)
    
    # Read the file
    df = read_data(f's3://{bucket_name}/{file_name}')
    
    # Assert the data is read correctly
    assert not df.empty
    # Additional assertions can be added as needed

