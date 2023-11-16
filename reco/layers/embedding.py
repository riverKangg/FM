import torch
import torch.nn as nn

class FeatureEmbedding(nn.Module):
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
        Initializes the FeatureEmbedding module.

        Args:
            featuremap (dict): A dictionary containing feature information.
            embedding_size (int): The size of the embedding vectors.

        """
        super(FeatureEmbedding, self).__init__()
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

    embedding_model = FeatureEmbedding(featuremap, embedding_size=5)
    embedded_data = embedding_model(input_data)

    print("Input Data Shapes:")
    for feat in input_data:
        print(f"{feat}: {input_data[feat].shape}")

    print("\nEmbedded Data Shape:")
    print(embedded_data.shape)
