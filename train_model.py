import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def scale_frame(frame):
    df = frame.copy()
    # предсказываем salary_in_usd, удаляем лишние текстовые колонки зарплат
    X = df.select_dtypes(include=[np.number]).drop(columns=['salary_in_usd', 'salary'], errors='ignore')
    y = df['salary_in_usd']
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X.values)
    return X_scale, y.values, scaler

if __name__ == "__main__":
    df = pd.read_csv("./df_clear.csv")
    X_scale, Y, scaler = scale_frame(df)
    X_train, X_val, y_train, y_val = train_test_split(X_scale, Y, test_size=0.3, random_state=42)
    
    mlflow.set_experiment("salary model")
    with mlflow.start_run():
        lr = SGDRegressor(random_state=42)
        lr.fit(X_train, y_train)
        
        y_pred = lr.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(lr, "model")

    # получаем путь к лучшей модели
    dfruns = mlflow.search_runs()
    path2model = dfruns.iloc[0]['artifact_uri'].replace("file://","") + '/model'
    print(path2model) # этот вывод пойдет в best_model.txt
