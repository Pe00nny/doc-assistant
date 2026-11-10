import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from vectorize import get_prepared_data
from model import TextClassifier

import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import json

def train():
    print("Инициализация обучения")

    # Загрузка подготовленных тензоров из vectorize.py
    X,y = get_prepared_data()
    in_features = X.shape[1]
    num_classes = int(y.max()) + 1


    X_tr, X_te, y_tr, y_te = train_test_split(
        X.numpy(), y.numpy(), test_size=0.2, random_state=42
    )

    X_train_tensor = torch.tensor(X_tr, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_tr, dtype=torch.long)
    X_test_tensor = torch.tensor(X_te, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_te, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = TextClassifier(in_features=in_features, num_classes=num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    best_test_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            correct_train += (predicted == batch_y).sum().item()
            total_train += batch_y.size(0)

            avg_train_loss = total_train_loss / len(train_loader)
            train_acc = correct_train / total_train

            model.eval()
            total_test_loss = 0
            correct_test = 0
            total_test = 0

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    logits = model(batch_X)
                    loss = loss_fn(logits, batch_y)

                    total_test_loss += loss.item()
                    _, predicted = torch.max(logits, dim=1)
                    correct_test += (predicted == batch_y).sum().item()
                    total_test += batch_y.size(0)

            avg_test_loss = total_test_loss / len(test_loader)
            test_acc = correct_test/ total_test

            print(f"Эпоха [{epoch+1:02d}/{num_epochs}] | "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} |"
                  f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}")

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                os.makedirs('models', exist_ok=True)

                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model_state_dict(),
                    'optimizer_state_dict': optimizer_state_dict(),
                    'loss': best_test_loss
                }
                torch.save(checkpoint, 'models/best_text_classifier.pth')
                print(f"--> Чекпоинт сохранен")
                
mlflow.set_experiment("classic-doc-classifier")
def train_and_log(lr, epochs, batch_size=32):
    print(f"Запуск эксперимента: lr={lr}, epochs={epochs}")
    
    X, y = get_prepared_data()
    in_features = X.shape[1]
    num_classes = int(y.max()) + 1
    
    X_tr, X_te, y_tr, y_te = train_test_split(
        X.numpy(), y.numpy(), teat_size=0.2, random_state=42
    )
    
    X_train_tensor = torch.tensor(X_tr, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_tr, dtype=torch.long)
    X_test_tensor = torch.tensor(X_te, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_te, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = TextClassifier(in_features=in_features, num_classes=num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history_train_loss = []
    history_test_loss = []
    
    with mlflow.start_run():
        mlflow.log_param("learning rate", lr)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("in_features", in_features)
        
        mlflow.set_tag("model_type", "TF-IDF + PyTorch NN")
        mlflow.set_tag("author", "Ваше имя")
        
        best_test_loss = float('inf')
        final_test_acc = 0.0
        
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                
            avg_train_loss = total_train_loss / len(train_loader)
            history_train_loss.append(avg_train_loss)
            
            model.eval()
            total_test_loss = 0
            correct_test= 0
            total_test = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    logits = model(batch_X)
                    loss = loss_fn(logits, batch_y)
                    total_test_loss += loss.item()
                    _, predicted = torch.max(logits, dim=1)
                    correct_test += (predicted == batch_y).sum().item()
                    total_test += batch_y.size(0)
                    
            avg_test_loss = total_test_loss / len(test_loader)
            test_acc = correct_test / total_test
            history_test_loss.append(avg_test_loss)
            
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("test_loss", avg_test_loss, step=epoch)
            mlflow.log_metric("test_acc", test_acc, step=epoch)
            
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                os.makedirs('models', exist_ok=True)
                torch.save(model.state_dict(), 'models/best_text_classifier.pth')
                
            final_test_acc = test_acc
            
        plt.figure(figure=(8, 5))
        plt.plot(range(1, epochs + 1), history_train_loss, label='Train_loss')
        plt.plot(range(1, epochs + 1), history_test_loss, label='Test_loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.title(f'Loss curve (lr={lr})')
        plt.legend()
        fig = plt.gcf()
        mlflow.log_figure(fig, "artifacts/loss_curve.png")
        plt.close()
        
        config = {"lr": lr, "epochs": epochs, "optimizer": "Adam"}
        with open("config.json", "w") as f:
            json.dump(config, f)
        mlflow.log_artifact("config.json")
        os.remove("config.json")
        
        mlflow.pytorch.log_model(model, "model")
        
        print(f"Эксперимент завершен. Финальная точность: {final_test_acc:.4f}")
        
def main():
    print("Инициализация серии экспериментов")
    params_grid = [
        {"lr": 1e-2, "epochs": 10},
        {"lr": 1e-3, "epochs": 15},
        {"lr": 1e-4, "epochs": 10},
        {"lr": 1e-3, "epochs": 10}
        
    ]
    
    for params in params_grid:
        train_and_log(lr=params["lr"], epochs=params["epochs"])
        
    print("Все эксперименты завершены")
            
        
        
        
if __name__ == "__main__":
    main()