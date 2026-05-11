import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

def scale_frame(frame):
    df = frame.copy()
    # удаляем текстовые и лишние колонки, оставляем 'salary' как цель
    X = df.drop(columns=['salary'])
    y = df['salary']
    
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X.values)
    return X_scale, y.values, scaler

if __name__ == "__main__":
    df = pd.read_csv("./df_clear.csv")
    # оставляем только числовые колонки
    df = df.select_dtypes(include=[np.number])
    
    X_scale, Y, scaler = scale_frame(df)
    X_train, X_val, y_train, y_val = train_test_split(X_scale, Y, test_size=0.3, random_state=42)
    
    mlflow.set_experiment("salary_model")
    with mlflow.start_run():
        lr = SGDRegressor(random_state=42)
        lr.fit(X_train, y_train)
        
        y_pred = lr.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(lr, "model")
        print(f"MAE: {mae}")

    # путь к модели для деплоя
    dfruns = mlflow.search_runs(experiment_names=["salary_model"])
    path2model = dfruns.iloc[0]['artifact_uri'].replace("file://","") + '/model'
    print(path2model)
