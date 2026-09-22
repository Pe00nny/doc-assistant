import os
from pathlib import Path
import pandas as pd

# 1. Создаем папку для сохранения результата
os.makedirs("data/processed", exist_ok=True)

# 2. Автоматически находим правильный путь к Загрузкам без кириллицы
downloads_dir = Path.home() / "Downloads"
path_to_file = downloads_dir / "train-00000-of-00001.parquet"

# Проверяем, существует ли файл перед чтением
if not path_to_file.exists():
    raise FileNotFoundError(f"Файл не найден по пути: {path_to_file}")

# 3. Читаем parquet-файл
print(f"Пытаюсь прочитать файл: {path_to_file}")
df = pd.read_parquet(path_to_file)

# 4. Очистка и фильтрация данных
df = df[["text", "topic"]].dropna()

# Удаляем переносы строк (\n, \r) из самого текста, чтобы Excel не дробил одну строку на несколько
df["text"] = df["text"].astype(str).str.replace(r"\r+|\n+", " ", regex=True)

# Перемешиваем датасет
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Строго отсекаем первые 1000 строк
df = df.head(1000)

# 5. Безопасное сохранение результата
output_file = "data/processed/clean.csv"

# Удаляем старый файл, если он существовал, чтобы данные не наложились
if os.path.exists(output_file):
    os.remove(output_file)

# Сохраняем (utf-8-sig идеально подходит для открытия в Excel на Windows)
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\n--- Готов! ---")
print(f"Итоговый файл успешно сохранен в: {output_file}")
print(f"Точное количество строк в датафрейме: {len(df)}")
print("\nРаспределение по темам:")
print(df["topic"].value_counts())
