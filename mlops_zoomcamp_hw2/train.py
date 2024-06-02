import os
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def train_and_evaluate(data_path):
    X_train, y_train = load_pickle(os.path.join(data_path, 'train.pkl'))
    X_val, y_val = load_pickle(os.path.join(data_path, 'val.pkl'))

    # Enable autologging
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        rf = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=2, 
            random_state=42
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        
        # Log metrics
        mlflow.log_metric('rmse', rmse)
        
        print(f'RMSE: {rmse}')

if __name__ == '__main__':
    data_path = './output'  # Adjust the path if necessary
    train_and_evaluate(data_path)
