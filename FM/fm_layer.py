import torch
import torch.nn as nn
from FM import *


class FMLayer(nn.Module):
    def __init__(self, featuremap):
        super(FMLayer, self).__init__()
        self.lr = LinearLayer(featuremap)
        self.inner_product = InnerProduct(featuremap)

    def forward(self, X, X_emb):
        output = self.lr(X) + self.inner_product(X_emb)
        return output


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

    embedding_layer = EmbeddingDict(featuremap)
    input_data_emb = embedding_layer(input_data)

    fm_layer = FMLayer(featuremap)
    output = fm_layer(input_data, input_data_emb)

    print("Output Shape:", output.shape)
    # print("Output Values:", output)