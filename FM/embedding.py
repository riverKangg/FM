import os
import json
import torch
import torch.nn as nn

os.chdir('/FM')


class EmbeddingDict(nn.Module):
    """
    Custom PyTorch module for creating an embedding dictionary.

    Args:
        featuremap (dict): A dictionary containing feature information.
        embedding_size (int): The size of the embedding vectors.

    Attributes:
        embedding_layer (nn.ModuleDict): A dictionary of nn.Embedding layers for each feature.

    Methods:
        forward(X): Performs forward pass to embed input data.

    """
    def __init__(self, featuremap, embedding_size=5):
        """
        Initializes the EmbeddingDict module.

        Args:
            featuremap (dict): A dictionary containing feature information.
            embedding_size (int): The size of the embedding vectors.

        """
        super(EmbeddingDict, self).__init__()
        feature_dict = featuremap['features']
        self.embedding_layer = nn.ModuleDict(
            {feat: nn.Embedding(feature_dict[feat]['vocab_size'], embedding_size) for feat in feature_dict}
        )

    def forward(self, X):
        """
        Performs forward pass to embed input data.

        Args:
            X (dict): Input data dictionary where keys are feature names and values are feature tensors.

        Returns:
            torch.Tensor: The concatenated tensor of embedded features.

        """
        feature_emb_dict = {feat: self.embedding_layer[feat](X[feat]) for feat in self.embedding_layer}
        feature_emb_list = []
        for feat in feature_emb_dict:
            feature_emb_list.append(feature_emb_dict[feat])
        feature_emb = torch.stack(feature_emb_list, dim=1)

        return feature_emb


if __name__ == "__main__":
    with open("../input/FeatureMap/avazu.json", 'rb') as json_file:
        feature_map = json.load(json_file)

    data = torch.load("../input/avazu/avazu.pt")
    X = data['X']

    emb = EmbeddingDict(feature_map)
    X_emb = emb(X)
    print(X_emb.shape)
