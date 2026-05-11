import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def download_data():
    url = "https://raw.githubusercontent.com/tpush/salary_train_pipe/refs/heads/main/job_salary_prediction_data.csv"
    df = pd.read_csv(url)
    df.to_csv("salary_data.csv", index=False)
    print("Data downloaded successfully.")
    return df

def clear_data(path):
    df = pd.read_csv(path)
    
    # удаляем строки с пустыми значениями в зарплате
    df = df.dropna(subset=['salary_in_usd'])
    
    # выбираем категориальные признаки для кодирования
    cat_columns = ['experience_level', 'employment_type', 'job_title', 'company_size']
    
    # преобразуем текст в числа
    ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[cat_columns] = ordinal.fit_transform(df[cat_columns].astype(str))
    
    # сохраняем очищенный файл
    df.to_csv('df_clear.csv', index=False)
    print("Data cleared and saved to df_clear.csv")

if __name__ == "__main__":
    download_data()
    clear_data("salary_data.csv")
