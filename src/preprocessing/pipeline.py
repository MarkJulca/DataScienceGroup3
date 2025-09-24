from extract import extract_kaggle_data

def generate_kaggle_data():
    gold_data = extract_kaggle_data()
    #parquet
    gold_data.to_parquet("data/raw/raw_kaggle_data.parquet")
    #csv
    gold_data.to_csv("data/raw/raw_kaggle_data.csv", index=False)
    return "success"

generate_kaggle_data()
