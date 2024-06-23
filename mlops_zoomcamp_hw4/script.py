import pickle
import pandas as pd
import numpy as np
import argparse

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main(year, month):
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = read_data(url)
    
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    # Create ride_id column
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    # Prepare the results dataframe
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })

    # Save the results to a parquet file
    output_file = 'predictions.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    
    # Print the mean predicted duration
    mean_duration = np.mean(y_pred)
    print(f'Mean predicted duration: {mean_duration:.2f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, help="Year of the data")
    parser.add_argument("--month", type=int, required=True, help="Month of the data")
    args = parser.parse_args()

    main(args.year, args.month)

