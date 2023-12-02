import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt

# Wczytaj dane
BostonHousing2 = load_breast_cancer()
data = BostonHousing2.data
target = LabelEncoder().fit_transform(BostonHousing2.target)

# Przygotuj dane
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=10)

# Zdefiniuj funkcję target
def target(x):
    return torch.eye(len(np.unique(x)))[x]

y_train_target = target(y_train)

# Definiuj model sieci neuronowej w PyTorch
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Inicjalizuj model
input_size = X_train.shape[1]
hidden_size = 6
output_size = len(np.unique(target))
model = NeuralNetwork(input_size, hidden_size, output_size)

# Definiuj funkcję straty i optymalizator
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trenuj model
num_epochs = 800
losses = []

for epoch in range(num_epochs):
    inputs = torch.from_numpy(X_train[:, 1:9]).float()
    labels = torch.from_numpy(y_train).long()

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

# Wykres funkcji straty
plt.plot(losses, color='darkred')
plt.xlabel("Iteracja")
plt.ylabel("Strata")
plt.show()

# Przewidywanie dla danych testowych
with torch.no_grad():
    inputs = torch.from_numpy(X_test[:, 1:9]).float()
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)

# Funkcja do oceny klasyfikacji
def test_klasyf(zad, wy):
    print(np.histogram2d(zad, wy, bins=(len(np.unique(zad)), len(np.unique(wy))))[0])

# Test klasyfikacji
test_klasyf(y_test, predicted.numpy())

# Dokładność klasyfikacji
accuracy = np.sum(np.diag(np.histogram2d(y_test, predicted.numpy(), bins=(len(np.unique(y_test)), len(np.unique(predicted.numpy()))))[0])) / len(y_test) * 100
print(f"Dokonano klasyfikacji: {accuracy:.2f}%")
