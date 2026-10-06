import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# 1. Создаем папку для сохранения результата
os.makedirs("data/processed", exist_ok=True)

# 2. Скачиваем чистый датасет с Hugging Face
DATASET_NAME = "data-silence/rus_news_classifier"

# Категории из карточки датасета: числовая метка -> название темы
categories_translator = {
    0: "climate",
    1: "conflicts",
    2: "culture",
    3: "economy",
    4: "gloss",
    5: "health",
    6: "politics",
    7: "science",
    8: "society",
    9: "sports",
    10: "travel",
}

print(f"Скачиваю датасет: {DATASET_NAME}")
ds = load_dataset(DATASET_NAME, split="train")

# 3. Переводим в DataFrame и приводим колонки к нужному виду
df = ds.to_pandas()
df = df.rename(columns={"news": "text", "labels": "topic"})
df["topic"] = df["topic"].map(categories_translator)

# 4. Очистка и фильтрация данных
df = df[["text", "topic"]].dropna()

# Удаляем переносы строк (\n, \r), чтобы Excel не дробил одну строку на несколько
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