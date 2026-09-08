import pandas as pd
import os

def main():
    print("Запуск пайплайна очистки данных...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, "..", "data", "raw", "data.csv")
    output_dir = "data/processed"
    ouput_path = os.path.join(output_dir, "clean.csv")
    
    # Проверяем наличие папки для сохранения
    os.makedirs(output_dir, exist_ok=True)
    
    # Шаги обработки
    df = pd.read_csv(input_path)
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"])
    df["text"] = df["text"].str.strip().str.replace(r"\s+", " ", regex=True)
    df = df[df["text"].str.len() > 50]
    
    # Сохранение результата
    df.to_csv(ouput_path, index=False)
    print(f"Готово: {len(df)} документов -> {ouput_path}")
    
if __name__ == "__main__":
    main()