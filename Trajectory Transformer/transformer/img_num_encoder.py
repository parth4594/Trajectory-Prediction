import torch.nn as nn
from torch.nn.functional import relu


class Image_Num_Encoder(nn.Module):
    def __init__(self, n_features, n_encodings, n_hidden, dropout=0.1):
        super(Image_Num_Encoder, self).__init__()
        self.w_1 = nn.Linear(n_features, n_hidden)
        self.w_2 = nn.Linear(n_hidden, n_encodings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.w_2(self.dropout(relu(self.w_1(x))))
        return output