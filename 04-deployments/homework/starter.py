import pickle
import pandas as pd
import numpy as np
import sys


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)
2023

categorical = ['PULocationID', 'DOLocationID']

def read_score_data(year, month):
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    
    return y_pred

#df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')


def run():
    year = int(sys.argv[1]) # 2021
    month = int(sys.argv[2]) # 3
    predictions = read_score_data(year,month)
    print('rediction_mean: ', np.mean(predictions))


if __name__ == '__main__':
    run()





