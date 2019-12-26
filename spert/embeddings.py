import torch
from torch import nn as nn

class CharacterEmbeddings(nn.Embedding):
    """Character embeddings of words, as proposed in Lample et al., 2016."""

    def __init__(
        self,
        char_embedding_dim: int = 25,
        hidden_size_char: int = 25,
    ):
        """Uses the default character dictionary if none provided."""

        super().__init__()

        # use list of common characters if none provided

        self.char_embedding_dim = char_embedding_dim
        self.hidden_size_char = hidden_size_char
        self.char_embedding = nn.Embedding(
            len(self.char_dictionary.item2idx), self.char_embedding_dim
        )
        self.char_rnn = torch.nn.LSTM(
            self.char_embedding_dim,
            self.hidden_size_char,
            num_layers=1,
            bidirectional=True,
        )

        self.__embedding_length = self.hidden_size_char * 2
