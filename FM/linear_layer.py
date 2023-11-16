import torch
import torch.nn as nn
from FM.embedding import EmbeddingDict


class LinearLayer(nn.Module):
    def __init__(self, featuremap, use_bias=True):
        super(LinearLayer, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True) if use_bias else None
        self.embedding = EmbeddingDict(featuremap, embedding_size=1)

    def forward(self, X):
        y_pred = self.embedding(X).sum(dim=1)
        if self.bias is not None:
            y_pred += self.bias
        return y_pred


if __name__ == "__main__":
    featuremap = {
        'features': {
            'feature1': {'vocab_size': 5},
            'feature2': {'vocab_size': 10},
            'feature3': {'vocab_size': 3},
        }
    }

    input_data = {
        'feature1': torch.randint(0, 5, (1000,)),
        'feature2': torch.randint(0, 10, (1000,)),
        'feature3': torch.randint(0, 3, (1000,))
    }

    lr_layer = LinearLayer(featuremap)
    output = lr_layer(input_data)
    print("Output Shape:", output.shape)
