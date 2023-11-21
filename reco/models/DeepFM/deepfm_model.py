import torch
import torch.nn as nn
from reco.layers.embedding import FeatureEmbedding


class DeepFM(nn.Module):
    def __init__(self, featuremap):
        self.embeddings = FeatureEmbedding()
