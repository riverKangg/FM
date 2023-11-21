import torch
import torch.nn as nn
from reco.layers.fm_layer import FMLayer
from reco.layers.dnn_layer import DNNLayer
from reco.layers.embedding import FeatureEmbedding


class DeepFM(nn.Module):
    def __init__(self, featuremap, embedding_size=5):
        super(DeepFM, self).__init__()
        _input_dim = embedding_size * len(featuremap["features"])
        self.embeddings = FeatureEmbedding(featuremap, embedding_size)
        self.fmlayer = FMLayer(featuremap)
        self.dnnlayer = DNNLayer(_input_dim)

    def forward(self, X, y):
        X_emb = self.embeddings(X)
        fm = self.fmlayer(X, X_emb)

        dnn = self.dnnlayer(X)

        y_pred = torch.sigmoid(fm + dnn)
        y = y.reshape(y_pred.shape)
        return {'y_true': y, 'y_pred': y_pred}


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

    label = torch.randint(0, 1, (1000,))

    model = DeepFM(featuremap)
    output = model(input_data, label)
    print(output.size)
