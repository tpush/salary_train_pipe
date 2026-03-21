import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
from pathlib import Path
import os
from datetime import timedelta
from train_model import train

def download_data():
    df = pd.read_csv('job_salary_prediction_dataset.csv')
    df.to_csv("jobs.csv", index = False)
    print("df: ", df.shape)
    return df

def clear_data():
    df = pd.read_csv("jobs.csv")
    
    cat_columns = ['job_title', 'education_level', 'industry', 'company_size', 'location', 'remote_work']
    num_columns = ['experience_years', 'skills_count', 'certifications', 'salary']
    
    # Анализ и очистка данных
    # Удаление записей с отрицательным или нулевым опытом
    question_exp = df[df.experience_years < 0]
    df = df.drop(question_exp.index)
    
    # Анализ аномалий зарплат (например, слишком низкие значения)
    question_salary_low = df[df["salary"] < 1000]
    df = df.drop(question_salary_low.index)
    
    # Ограничение по максимальной зарплате (аналог анализа гистограмм)
    question_salary_high = df[df["salary"] > 1000000]
    df = df.drop(question_salary_high.index)
    
    # Ограничение по количеству навыков
    question_skills = df[df.skills_count > 50]
    df = df.drop(question_skills.index)
    
    df = df.reset_index(drop=True)  
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns])
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    df.to_csv('df_clear.csv', index=False)
    return True

dag_jobs = DAG(
    dag_id="salary_train_pipe",
    start_date=datetime(2025, 2, 3),
    schedule=timedelta(minutes=5),
    max_active_runs=1,
    catchup=False,
)

download_task = PythonOperator(python_callable=download_data, task_id = "download_jobs", dag = dag_jobs)
clear_task = PythonOperator(python_callable=clear_data, task_id = "clear_jobs", dag = dag_jobs)
train_task = PythonOperator(python_callable=train, task_id = "train_jobs", dag = dag_jobs)

download_task >> clear_task >> train_task