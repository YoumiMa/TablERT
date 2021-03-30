import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, input_dim, hid_dim, device):
        ''' We borrow the idea of multi-head self-attention,
        but our goal is not to compute attention weights over entity representations. 
        We only use parameters of our concern here, i.e., Q and K. 
        '''

        super(MultiHeadAttention, self).__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        self.w_q = nn.Linear(input_dim, hid_dim * n_heads)
        self.w_k = nn.Linear(input_dim, hid_dim * n_heads)

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)

        Q = Q.view(batch_size, -1, self.n_heads, self.hid_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.hid_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0,1,3,2))

        if mask is not None:
            energy = energy.masked_fill(mask==0, -1e10)

        attention = energy



        return attention

