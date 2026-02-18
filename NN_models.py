import torch
import torch.nn as nn
import torch.optim as optim


class NN_MLP(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512):
        super(NN_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.sigmoid(out)
    

def build_mlp():
    model = NN_MLP()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    return model, optimizer, criterion, epochs


MODEL_REGISTRY = {
    "NN_MLP": build_mlp}