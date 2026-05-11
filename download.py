import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def download_data():
    url = 'https://raw.githubusercontent.com/tpush/salary_train_pipe/refs/heads/main/job_salary_prediction_data.csv'
    df = pd.read_csv(url, delimiter = ',')
    df.to_csv("salary.csv", index = False)
    return df

def clear_data(path2df):
    df = pd.read_csv(path2df)
    
    # колонки для кодирования
    cat_columns = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']
    
    # очистка данных
    # удаляем пустые значения в целевой переменной
    df = df.dropna(subset=['salary_in_usd'])
    
    # сбрасываем индексы и кодируем категории в числа
    df = df.reset_index(drop=True)  
    ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    ordinal.fit(df[cat_columns])
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    
    df.to_csv('df_clear.csv', index=False)
    return True

if __name__ == "__main__":
    download_data()
    clear_data("salary.csv")
