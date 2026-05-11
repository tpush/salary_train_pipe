import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def download_data():
    url = 'https://raw.githubusercontent.com/tpush/salary_train_pipe/refs/heads/main/job_salary_prediction_data.csv'
    df = pd.read_csv(url, delimiter = ',')
    df.to_csv("salary.csv", index = False)
    print("Data downloaded")
    return df

def clear_data(path2df):
    df = pd.read_csv(path2df)
    
    # Очистка данных
    df = df.dropna(subset=['salary'])
    
    # категориальные колонки
    cat_columns = ['job_title', 'education_level', 'industry', 'company_size', 'location', 'remote_work']
    
    # кодирование категорий
    df = df.reset_index(drop=True)  
    ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    existing_cats = [col for col in cat_columns if col in df.columns]
    
    df[existing_cats] = ordinal.fit_transform(df[existing_cats].astype(str))
    
    # сохраняем очищенный файл
    df.to_csv('df_clear.csv', index=False)
    print("Data cleared and saved to df_clear.csv")
    return True

if __name__ == "__main__":
    download_data()
    clear_data("salary.csv")
