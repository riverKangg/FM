import torch
import torch.nn as nn
from reco.layers.fm_layer import FMLayer
from reco.layers.embedding import FeatureEmbedding


class FM(nn.Module):
    def __init__(self, featuremap):
        super(FM, self).__init__()
        self.embedding = FeatureEmbedding(featuremap)
        self.fm = FMLayer(featuremap)

    def forward(self, X, y):
        X_emb = self.embedding(X)
        y_pred = self.fm(X, X_emb)
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

    model = FM(featuremap)
    output = model(input_data, label)
    print(output['y_pred'].shape, output['y_true'].shape)
