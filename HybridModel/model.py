import torch
from torch import nn
import torchvision
from torchvision import transforms
import numpy as np

class CNNBlock(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.input_shape = input_shape
        self.Layer = torch.nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size= (3,3)),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size= (3,3)),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= (2,2)),
            nn.Conv2d(hidden_units, output_shape, kernel_size=(5,5)),
            nn.BatchNorm2d(output_shape),
            nn.ReLU()
        )

    def get_output_shape(self, input_height, input_width):
        x = torch.randn(1, self.input_shape, input_height, input_width)
        return self.Layer(x).shape[2:]

    def forward(self, x):
        return self.Layer(x)
    
class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels: int,
                 patch_size: int,
                 embedding_dim: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.patcher = nn.Conv2d(in_channels= in_channels,
                                 out_channels= embedding_dim,
                                 stride= patch_size,
                                 kernel_size= patch_size,
                                 padding= 0)
        self.flatten = nn.Flatten(start_dim= 2,
                                  end_dim= 3)

    def forward(self, x):
        image_res = x.shape[-1]
        assert (image_res % self.patch_size == 0), "patch size should be divisible with image resolution"
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0,2,1)
    
class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim : int,
                 num_heads : int,
                 att_dropout : float):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(normalized_shape= embedding_dim)

        self.MultiHeadAttention = nn.MultiheadAttention(embed_dim= embedding_dim,
                                                        num_heads= num_heads,
                                                        dropout= att_dropout,
                                                        batch_first= True)

    def forward(self, x):
        x = self.LayerNorm(x)
        attn_output, _ = self.MultiHeadAttention(query= x,
                                                 key= x,
                                                 value= x,
                                                 need_weights = False)
        return attn_output
    
class MultiLayerPreceptronBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 mlp_size: int,
                 dropout: float):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(normalized_shape= embedding_dim)

        self.MLP = nn.Sequential(
            nn.Linear(in_features= embedding_dim,
                      out_features= mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features= mlp_size,
                      out_features= embedding_dim),
            nn.Dropout(p= dropout)
        )

    def forward(self, x):
        x = self.LayerNorm(x)
        x = self.MLP(x)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 mlp_size: int,
                 attn_dropout: float,
                 mlp_dropout: float):
        super().__init__()
        self.MSA_Block = MultiHeadSelfAttentionBlock(embedding_dim= embedding_dim,
                                               num_heads= num_heads,
                                               att_dropout= attn_dropout)
        self.MLP_Block = MultiLayerPreceptronBlock(embedding_dim= embedding_dim,
                                             mlp_size= mlp_size,
                                             dropout= mlp_dropout)

    def forward(self, x):
        x = self.MSA_Block(x) + x
        x = self.MLP_Block(x) + x
        x = self.MSA_Block(x) + x
        return x
    
class ViTBlock(nn.Module):
    def __init__(self,
                 image_size: int,
                 in_channels: int,
                 patch_size: int,
                 num_transformer_layers: int,
                 embedding_dim: int,
                 mlp_size: int,
                 num_heads: int,
                 attn_dropout: float,
                 mlp_dropout: float,
                 embedding_dropout: float,
                 num_classes: int = 2):
        super().__init__()

        assert image_size % patch_size == 0, "patch size is divisible by image size"

        self.num_patches = (image_size // patch_size) ** 2

        self.class_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim),
                                            requires_grad= True)

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim),
                                               requires_grad= True)

        self.patch_embedding = PatchEmbedding(in_channels= in_channels,
                                              patch_size= patch_size,
                                              embedding_dim= embedding_dim)

        self.embedding_dropout = nn.Dropout(p = embedding_dropout)

        self.transformerencoder = nn.Sequential(* [TransformerEncoder(embedding_dim= embedding_dim,
                                                     num_heads= num_heads,
                                                     mlp_size= mlp_size,
                                                     attn_dropout= attn_dropout,
                                                     mlp_dropout= mlp_dropout) for _ in range(num_transformer_layers)])


    def forward(self, x):
        batch_size = x.shape[0]

        class_token = self.class_embedding.expand(batch_size, -1, -1)

        x = self.patch_embedding(x)

        x = torch.cat((class_token, x), dim = 1)

        x = self.position_embedding + x

        x = self.embedding_dropout(x)

        x = self.transformerencoder(x)

        return x
    
class AttentionMechBlock(nn.Module):
    def __init__(self, dim, units=128):
        super().__init__()
        self.query = nn.Linear(dim, units)
        self.key = nn.Linear(dim, units)
        self.value = nn.Linear(dim, units)
        self.LayerNorm = nn.LayerNorm(normalized_shape= units)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = torch.softmax(Q @ K.transpose(1,2) / (x.size(-1)**0.5), dim=-1)
        return self.LayerNorm((attn @ V).mean(dim=1))

class HybridModel(nn.Module):

    def __init__(self,
                 image_size: int,
                 in_channels: int,
                 hidden_units: int,
                 output_shape: int,
                 patch_size: int,
                 num_transformer_layers: int,
                 embedding_dim: int,
                 mlp_size: int,
                 num_heads: int,
                 attn_dropout: float,
                 mlp_dropout: float,
                 embedding_dropout: float,
                 units: int = 128,
                 num_classes: int = 2):
        super().__init__()
        self.CNNBlock = CNNBlock(input_shape= 3,
                                 hidden_units= hidden_units,
                                 output_shape= output_shape)
        self.cnn_output_height, self.cnn_output_width = self.CNNBlock.get_output_shape(image_size, image_size)
        self.ViTBlock = ViTBlock(image_size= self.cnn_output_height,
                                 in_channels= in_channels,
                                 patch_size= patch_size,
                                 num_transformer_layers= num_transformer_layers,
                                 embedding_dim= embedding_dim,
                                 mlp_size= mlp_size,
                                 num_heads= num_heads,
                                 attn_dropout= attn_dropout,
                                 mlp_dropout= mlp_dropout,
                                 embedding_dropout= embedding_dropout,
                                 num_classes= num_classes)
        self.AttentionMechBlock = AttentionMechBlock(dim= embedding_dim,
                                                     units= units)
        self.classifier = torch.nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p = mlp_dropout),
            nn.Linear(in_features= units,
                      out_features= num_classes)
        )

    def forward(self, x):
        x = self.CNNBlock(x)
        x = self.ViTBlock(x)
        x = self.AttentionMechBlock(x)
        x = self.classifier(x)
        return x