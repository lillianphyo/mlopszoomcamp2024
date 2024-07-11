import os
import pandas as pd
import subprocess
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

# Create the test data
data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
]
columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

# Define the input file path
input_file = 's3://nyc-duration/in/2023-01.parquet'

# Set S3 endpoint URL
S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
options = {
    'client_kwargs': {
        'endpoint_url': S3_ENDPOINT_URL
    }
}

# Save the DataFrame to S3
df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

# Run the batch script
# subprocess.run(['python', 'homework/batch.py', '2023', '1'])
subprocess.run(['python', 'batch_Q6.py', '2023', '1'])

# Define the output file path
output_file = 's3://nyc-duration/out/2023-01.parquet'

# Read the result data from S3
df_result = pd.read_parquet(output_file, storage_options=options)

# Verify the result
predicted_duration_sum = df_result['predicted_duration'].sum()
print('Sum of predicted durations:', predicted_duration_sum)
