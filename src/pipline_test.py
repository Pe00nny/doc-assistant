import torch
import torch.nn as nn
from vectorize import get_prepared_data
from model import TextClassifier

def main():
    print("Тестовый старт")
    
    X, y = get_prepared_data()
    
    in_features = X.shape[1] # Это будет 1000
    num_classes = int(y.max()) + 1 # Автоматически определяет точное кол-во уникальных тем новостей
    
    model = TextClassifier(in_features=in_features, num_classes=num_classes)
    print("\nСтруктура нейросети")
    print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Шаг А: прямой ход (Forward pass)
    logits = model(X)
    
    # Шаг Б: считаем ошибку (Loss)
    loss = loss_fn(logits, y)
    print("\n[ЗАМЕР] Значение ошибки (Loss) ДО шага обучения: ", loss.item())
    
    # Шаг В: Обнуление градиентов
    optimizer.zero_grad()
    
    # Шаг Г: Обратный ход (Backward pass)
    loss.Backward()
    
    # Шаг Д: Корректировка весов (Step)
    optimizer.step()
    
    # 5. Контрольный замер после шага обучения
    new_logits = model(X)
    new_loss = loss_fn(new_logits, y)
    print("[ЗАМЕР] Значение ошибки (Loss) ПОСЛЕ шага обучения: ", new_loss.item())
    
    if new_loss.item() < loss.item():
        print("Веса скорректированы, ошибка модели пошла вниз")
    
if __name__ == "__main__":
    main()
