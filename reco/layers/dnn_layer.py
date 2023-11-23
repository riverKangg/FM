import torch
import torch.nn as nn


class DNNLayer(nn.Module):
    def __init__(self, input_dim,
                 hidden_units=[64, 64, 32],
                 output_dim=1,
                 use_bias=True,
                 dropout_rate=0.0):
        super(DNNLayer, self).__init__()
        dnn_layers = []
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dnn_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1],
                                        bias=use_bias))
            dnn_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(dropout_rate))
        dnn_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))

        self.dnn = nn.Sequential(*dnn_layers)

    def forward(self, input):
        if type(input) == dict:
            input = torch.stack([input[key] for key in input], dim=1)
        input = input.to(torch.float32)
        return self.dnn(input)


if __name__ == "__main__":
    input_dim = 10
    input_data = torch.rand((100, input_dim))
    dnn = DNNLayer(input_dim)
    output = dnn(input_data)
    print(input_data.size(), output.size())
