import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plot
import simFile

output_file = simFile.WriteToFile()

seed = 1992
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Załaduj dane Iris
iris = fetch_california_housing()
X = iris.data
y = iris.target

# Podział danych na zestawy treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Standaryzacja danych
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Konwersja danych do tensorów PyTorch
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)


# Definicja modelu
class Classifier(nn.Module):

    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        super().__init__()
        # 3 warstowoy model z funkcją aktywacji
        self.input_layer = nn.Linear(input_neurons, hidden_neurons)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(hidden_neurons, output_neurons)

    def forward(self, data):
        # Przejście przez warstwy z funkcją aktywacji
        data = self.input_layer(data)
        data = self.relu(data)
        data = self.hidden_layer(data)
        return data


# Inicjalizacja modelu, funkcji straty i optymalizatora
input_neurons = X_train.shape[1]
hidden_neurons = 60
#output_neurons = len(set(y_train))
output_neurons = 400
model = Classifier(input_neurons, hidden_neurons, output_neurons)

# Ustawienie seed'a dla modelu (np. inicjalizacja wag)
torch.manual_seed(seed)
model.apply(lambda x: torch.manual_seed(seed) if type(x) == nn.Linear else None)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

lose = []

# Trening modelu
num_epochs = 300
for epoch in range(num_epochs):
    # Forward pass
    output = model(X_train_tensor)
    # ^ == output = model.forward(X_train_tensor)
    # Obliczenie funkcji straty
    loss = criterion(output, y_train_tensor)

    # Backward pass and optimization
    # Wyzerowanie gradientów
    optimizer.zero_grad()
    # Obliczenie gradientów
    loss.backward()
    # Aktualizacja wag
    optimizer.step()

    lose.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoka [{epoch + 1}/{num_epochs}], Strata: {loss.item()}')
        output_file.epoch_update(f'Epoka [{epoch + 1}/{num_epochs}], Strata: {loss.item()}')

def plot_go():
    # Rysowanie wykresu straty
    plot.plot(range(1, num_epochs + 1), lose, label="Strata treningowa")
    plot.xlabel("Epoka")
    plot.ylabel("Strata")
    plot.title("Wykres straty podczas treningu")
    plot.legend()
    plot.show()


# Ocena modelu na zestawie testowym
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    # Wybór klasy z najwyższym prawdopodobieństwem
    _, predicted = torch.max(test_outputs, 1)
    # Obliczenie dokładności
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)

print(f'Dokładność modelu: {accuracy * 100:.2f}%')
output_file.result_update(f'Dokładność modelu: {accuracy * 100:.2f}%')

layers_of_neurons = 3
output_file.data_update(seed, layers_of_neurons, input_neurons,
                        hidden_neurons, output_neurons,
                        num_epochs)
output_file.layers_uprate(layers_of_neurons)
output_file.write_to_file()

plot_go()
