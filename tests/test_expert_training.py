#%env CUDA_VISIBLE_DEVICES=
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind import RemoteExpert
from test_utils.run_server import background_server

from sklearn.datasets import load_digits

def test_training(port:int):
    dataset = load_digits()
    X_train, y_train = torch.tensor(dataset['images'], dtype=torch.float), torch.tensor(dataset['target'])

    class Model(nn.Module):
        def __init__(self, expert1, expert2):
            super().__init__()
            self.fc = nn.Linear(64, 10)
            self.ex1 = expert1
            self.ex2 = expert2

        def forward(self, x):
            x = self.ex1(F.tanh(self.ex2(x)))
            x = self.fc(x)
            return x

    with background_server(num_experts=2, device='cpu', port=port, hidden_dim=64):
        expert1 = RemoteExpert('expert.0', host='127.0.0.1', port=port)
        expert2 = RemoteExpert('expert.1', host='127.0.0.1', port=port)
        print("experts connected")
        model = Model(expert1, expert2)
        print("model created")

        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(model.parameters(), lr=0.02)

        train_size = y_train.shape[0]

        for epoch in range(11):
            permutation = np.random.permutation(np.arange(train_size))
            X_train = X_train[permutation]
            y_train = y_train[permutation]

            opt.zero_grad()
            inputs = X_train
            labels = y_train

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
        train_pred = model(X_train).argmax(dim=1)
        accuracy = np.count_nonzero(train_pred == y_train)/y_train.shape[0]
        return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=1376, required=False)
    args = vars(parser.parse_args())
    accuracy = test_training(args.port)
    assert accuracy >= 0.9, "too small accuracy: {accuracy}"