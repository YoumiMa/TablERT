import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, input_dim, hid_dim, device):
        ''' general -> H_j^T W_a q
            dot -> H_j^T q
        '''

        super(MultiHeadAttention, self).__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(input_dim, hid_dim * n_heads)
        self.w_k = nn.Linear(input_dim, hid_dim * n_heads)
        self.w_v = nn.Linear(input_dim, hid_dim * n_heads)


        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()


    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.hid_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.hid_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.hid_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0,1,3,2))/self.scale

        if mask is not None:
            energy = energy.masked_fill(mask==0, -1e10)

        # attention = self.softmax(energy)
        attention = energy

        x = torch.matmul(attention, V)


        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim * self.n_heads)


        return attention, x

