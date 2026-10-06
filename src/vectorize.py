import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "clean.csv"

def get_prepared_data():
    df = pd.read_csv(DATA_PATH)

    vectorizer = TfidfVectorizer(max_features=1000)
    x_numpy = vectorizer.fit_transform(df["text"])

    label_encoder = LabelEncoder()
    y_numpy = label_encoder.fit_transform(df["topic"])

    x_tensor = torch.tensor(x_numpy.toarray(), dtype=torch.float32)
    y_tensor = torch.tensor(y_numpy, dtype=torch.long)

    print(f"Обнаружено классов (тем): {len(label_encoder.classes_)}")

    return x_tensor, y_tensor

if __name__ == "__main__":
    get_prepared_data()