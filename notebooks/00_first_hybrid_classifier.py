

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import math
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/ABC.csv')
feature_cols = ['Close']

# 1. Просто превращаем каждую строчку в % изменения относительно вчерашнего дня
# Метод .pct_change() встроен в Pandas и делает всё сам!
df_scaled = df[feature_cols].pct_change()

# 2. Считаем лог-доходность и таргет
df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
df['target'] = (df['log_returns'].shift(-1) > 0).astype(int)

# Склеиваем и чистим от первой пустой строки
df_scaled['target'] = df['target']
df_processed = df_scaled.dropna().reset_index(drop=True)

# 3. Нарезаем на 5-дневные окна
window_size = 5
X_list = []
y_list = []

features_array = df_processed[feature_cols].values
targets = df_processed['target'].values

for i in range(len(df_processed) - window_size):
    X_list.append(features_array[i : i + window_size])
    y_list.append(targets[i + window_size - 1])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)

y_hot = np.eye(2)[y].astype(np.float32)
X_train, X_test, y_hot_train, y_hot_test = train_test_split(X, y_hot, test_size=0.2, shuffle=False)


import pennylane as qml

n_qubits = 4
dev = qml.device("lightning.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs * torch.pi, wires=range(n_qubits), rotation="X") #переводит кубит в нужное состояние
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits)) #запутывает кубиты между собой
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)] #выдает число, определяющее мат ожидание класса с помощью оператора PauliZ


n_layers = 2
weight_shapes = {"weights": (n_layers, n_qubits)}


class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clayer_1 = torch.nn.Linear(5, 4)
        self.qlayer_1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer_2 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.clayer_2 = torch.nn.Linear(4, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x): # тензор проходит через нашу схему
        x = torch.flatten(x, start_dim=1)
        x = self.clayer_1(x)
        #x_1, x_2 = torch.split(x, 4, dim=1)
        x = self.qlayer_1(x)
        #x_2 = self.qlayer_2(x_2)
        #x = torch.cat([x_1, x_2], axis=1)
        x = self.clayer_2(x)
        return self.softmax(x)

model = HybridModel()


optimizer = torch.optim.SGD(model.parameters(), lr=0.007) # градиентный спуск
loss = torch.nn.L1Loss() # средний квадрат


X_train = torch.tensor(X_train, requires_grad=True).float()
y_hot_train = torch.tensor(y_hot_train).float()
X_test = torch.tensor(X_test, requires_grad=True).float()

batch_size = 10
batches = 200 // batch_size

data_loader_train = torch.utils.data.DataLoader(
    list(zip(X_train, y_hot_train)), batch_size=5, shuffle=True, drop_last=True
)
data_loader_test = torch.utils.data.DataLoader(
    list(zip(X_test, y_hot_test)), batch_size=5, shuffle=True, drop_last=True
)

epochs = 30


criterion = torch.nn.CrossEntropyLoss()
#print("Веса квантового слоя ДО:", model.qlayer_1.weights)
for epoch in range(epochs):
    running_loss = 0
    for batch_X, batch_y in data_loader_train:
        optimizer.zero_grad()  # Обнуляем прошлые градиенты
        
        # Пропускаем батч через модель. 
        # На выходе получаем тензор вероятностей размера [5, 2]
        outputs = model(batch_X) 
        
        # Считаем ошибку BCELoss между предсказанием [5, 2] и реальным batch_y [5, 2]
        loss_evaluated = criterion(outputs, batch_y)
        
        # Делаем магию PyTorch: считаем градиенты и делаем шаг обучения
        loss_evaluated.backward()
        optimizer.step()
    
        running_loss += loss_evaluated.item()

    y_pred_test = model(X_test)
    predictions = torch.argmax(y_pred_test, axis=1).detach().numpy()
    true_labels = torch.argmax(torch.tensor(y_hot_test), dim=1).numpy()
    correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, true_labels)]
    accuracy = sum(correct) / len(correct)
    print(f"Accuracy: {accuracy * 100}%")
    
    avg_loss = running_loss / len(data_loader_train)
    print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))
    #print("Веса квантового слоя ПОСЛЕ:", model.qlayer_1.weights)

y_pred_test = model(X_test)
predictions = torch.argmax(y_pred_test, axis=1).detach().numpy()
true_labels = torch.argmax(torch.tensor(y_hot_test), dim=1).numpy()
correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, true_labels)]
accuracy = sum(correct) / len(correct)
print(f"Accuracy: {accuracy * 100}%")




