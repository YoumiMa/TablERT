import torch
from torch import nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel
from transformers import BertTokenizer


from spert import sampling
from spert import util

from spert.embeddings import CharacterEmbeddings
from spert.entities import Token
from spert.loss import SpERTLoss, Loss
from spert.attention import MultiHeadAttention
from spert.beam import BeamSearch


from typing import List

# import torchcrf
import json
import itertools

import numpy


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h


def align_bert_embeddings(h: torch.tensor, token_mask: torch.tensor, cls_repr = None):
    """ Align bert token embeddings to word embeddings, masked by token_mask. """

    def pick_first(h: torch.tensor, token_mask: torch.tensor):
        """ pick up the first subword repr as word embedding."""
        
        sub_reprs = h[token_mask]
        return sub_reprs[0]

    def mean_pooling(h: torch.tensor, token_mask: torch.tensor):
        """ mean-pooling through bert embeddings. """
        
        sub_reprs = h[token_mask]
        avg_repr = sub_reprs.sum(dim=0)/sub_reprs.shape[0]

        return avg_repr

    def mean_pooling_with_cls(h: torch.tensor, token_mask: torch.tensor, cls_repr: torch.tensor):
        """ mean-pooling with cls. """
        
        sub_reprs = h[token_mask]
        avg_repr = (sub_reprs.sum(dim=0) + cls_repr.squeeze(0))/(sub_reprs.shape[0]+1)

        return avg_repr

    def max_pooling(h: torch.tensor, token_mask: torch.tensor):
        """ max-pooling through bert embeddings. """
        sub_reprs = h[token_mask]

        return sub_reprs.max(dim=0)[0]
    

    context_size = token_mask.shape[0]
    # print("context_size:", context_size)
    word_embeddings_lst = []

    for i in range(context_size):
        if torch.any(token_mask[i]):
            word_embeddings_lst.append(max_pooling(h, token_mask[i]))

    word_embeddings = torch.stack(word_embeddings_lst, dim=0)
    return word_embeddings


class TableF(BertPreTrainedModel):
    """ Span-based model to jointly extract entities and relations """

    def __init__(self, config: BertConfig, tokenizer: BertTokenizer,
                 relation_labels: int, entity_labels: int, beam_size: int,
                 entity_label_embedding: int,  att_hidden: int,
                 prop_drop: float, freeze_transformer: bool, device):
        super(TableF, self).__init__(config)

        # BERT model
        self.bert = BertModel(config)
        self._tokenizer = tokenizer
        self._device = device
        # layers
        self.entity_label_embedding = nn.Embedding(entity_labels , entity_label_embedding)
        self.entity_classifier = nn.Linear(config.hidden_size * 2, entity_labels)
       # sel.crf = torchcrf.CRF(entity_labels)
        self.attn = MultiHeadAttention(relation_labels, config.hidden_size + entity_label_embedding, att_hidden , device)
        self.dropout = nn.Dropout(prop_drop)

        self._relation_labels = relation_labels
        self._entity_labels = entity_labels
        self._beam_size = beam_size

        # weight initialization
        self.init_weights()

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def _get_prev_mask(self, prev_i, curr_i, ctx_size):
        # for word
        prev_mask = torch.zeros((ctx_size,), dtype=torch.bool)
        prev_mask[prev_i:curr_i] = 1

        return prev_mask


    def _get_prev_mask_(self, prev_i, curr_i, ctx_mask):
        # for token
        prev_mask = torch.zeros_like(ctx_mask, dtype=torch.bool)
        prev_mask[prev_i:curr_i] = 1

        return prev_mask


    def _forward_token(self, h: torch.tensor, curr_index: int, 
                prev_mask: torch.tensor, prev_embedding: torch.tensor):

        curr_repr = h[curr_index].unsqueeze(0)
        

        # maxpool between previous entity and current position.
        masked_repr = h[prev_mask]
        # print("masked repr:", masked_repr, masked_repr.shape)
        masked_repr_pool = masked_repr.max(dim=0)[0].unsqueeze(0)
#         cls_repr = h[0].unsqueeze(0)

        # previous label embedding.
        prev_embedding = prev_embedding.unsqueeze(0)

        # print(curr_repr.shape, masked_repr_pool.shape, prev_embedding.shape)
        # concat them for linear classification.
        entity_repr = torch.cat([curr_repr, masked_repr_pool, prev_embedding], dim=1)
        
        # dropout.
        entity_repr = self.dropout(entity_repr)


        entity_logits = self.entity_classifier(entity_repr)

        return entity_logits


    def _forward_token_rnn(self, h: torch.tensor, token_mask: torch.tensor, 
                           gold_seq: torch.tensor, entity_mask: torch.tensor):

        num_steps = gold_seq.shape[-1]
        word_h = h.repeat(token_mask.shape[0], 1, 1) * token_mask.unsqueeze(-1)
        word_h_pooled = word_h.max(dim=1)[0]
        word_h_pooled = word_h_pooled[:num_steps+2].contiguous()

        # curr word repr.
        curr_word_repr = word_h_pooled[1:-1].contiguous()
        # print("word_h:", word_h_pooled)
        # prev entity repr.
        prev_entity = torch.tril(entity_mask, diagonal=0)
        prev_entity[:, 0] = 0

        prev_entity_h = word_h_pooled.repeat(prev_entity.shape[0], 1, 1) * prev_entity.unsqueeze(-1)
        prev_entity_pooled = prev_entity_h.max(dim=1)[0]
        prev_entity_pooled = prev_entity_pooled[:num_steps].contiguous()
        # print("selected:", prev_entity_pooled)
        # prev_label_embedding.

        prev_seq = torch.cat([torch.tensor([0]).to(self._device), gold_seq])
        prev_label_embeddings = self.entity_label_embedding(prev_seq[:-1])
        # print("embedding:", prev_label_embeddings)

        rnn_input = torch.cat([self.dropout(curr_word_repr - 1), self.dropout(prev_entity_pooled - 1)], dim=1).unsqueeze(0)
#         rnn_input = self.dropout(curr_word_repr).unsqueeze(0) - 1
        # rnn_initial = torch.zeros((rnn_input.shape[0], rnn_input.shape[1], self._rnn_hidden_dim), dtype=torch.float).to(self._device)
        # rnn_output , hm = self.rnn(rnn_input, rnn_initial)

        # print("training:", rnn_input.shape)
        # curr_entity_logits = self.entity_classifier(self.dropout(rnn_output))
        curr_entity_logits = self.entity_classifier(rnn_input)
        # print("classifier grad:", self.entity_classifier.weight.grad)

        return curr_word_repr, curr_entity_logits


    def _forward_relation(self, h: torch.tensor,  entity_preds: torch.tensor, 
                          entity_mask: torch.tensor, gold_entity: torch.tensor,
                          is_eval: bool = False):
        if is_eval:
            entity_labels = entity_preds.unsqueeze(0)
        else:
            entity_labels = gold_entity.unsqueeze(0)

        # entity repr.
        masks_no_cls_rep = entity_mask[1:-1, 1:-1]
        entity_repr = h.repeat(masks_no_cls_rep.shape[-1], 1, 1) * masks_no_cls_rep.unsqueeze(-1)
        entity_repr_pool = entity_repr.max(dim=1)[0]

        # entity_label repr.
        entity_label_embeddings = self.entity_label_embedding(entity_labels)
        entity_label_repr = entity_label_embeddings.repeat(masks_no_cls_rep.shape[-1], 1, 1) * masks_no_cls_rep.unsqueeze(-1)
        entity_label_pool = entity_label_repr.max(dim=1)[0]


        rel_embedding = torch.cat([entity_repr_pool.unsqueeze(0) - 1, entity_label_pool.unsqueeze(0)], dim=2)
        rel_embedding = self.dropout(rel_embedding)
        att = self.attn(rel_embedding, rel_embedding, rel_embedding)

        return att

    def _forward_train(self, encodings: torch.tensor, context_mask: torch.tensor, 
                        token_mask: torch.tensor, start_labels: List[int], 
                        gold_entity: torch.tensor, entity_masks: List[torch.tensor],
                        allow_rel: bool):  
        # get contextualized token embeddings from last transformer layer
        context_mask = context_mask.float()
        h = self.bert(input_ids=encodings, attention_mask=context_mask)[0] + 1

        batch_size = encodings.shape[0]
        all_entity_logits = []
        all_rel_logits = []

        # print("entity masks:", entity_masks)
        for batch in range(batch_size): # every batch
            

            entity_mask = entity_masks[batch]

            word_h, curr_entity_logits = self._forward_token_rnn(h[batch], token_mask[batch], gold_entity[batch], entity_mask)

            entity_preds = torch.argmax(curr_entity_logits, dim=2)
            
            pred_entity_mask = torch.zeros_like(entity_mask, dtype=torch.bool).to(self._device).fill_diagonal_(1)

            # is_start = (entity_preds % 4 == 1 ) | (entity_preds % 4 == 2) | (entity_preds == 0)
            # print(is_start.shape)
            # print(~is_start)
            # not_start_mask = ~is_start.view()
            # print(torch.unique_consecutive(torch.ceil(entity_preds.float()/4),return_inverse=True))

            all_entity_logits.append(curr_entity_logits)
            # Relation classification.

            num_steps = gold_entity[batch].shape[-1]
            word_h = h[batch].repeat(token_mask[batch].shape[0], 1, 1) * token_mask[batch].unsqueeze(-1)
            word_h_pooled = word_h.max(dim=1)[0]
            word_h_pooled = word_h_pooled[:num_steps+2].contiguous()

            # curr word repr.
            curr_word_repr = word_h_pooled[1:-1].contiguous()

            curr_rel_logits = self._forward_relation(curr_word_repr, entity_preds.squeeze(0) , pred_entity_mask, gold_entity[batch])
            all_rel_logits.append(curr_rel_logits)

        if allow_rel:
            return all_entity_logits, all_rel_logits
        else:
            return all_entity_logits, []

    
    def _forward_eval(self, encodings: torch.tensor, context_mask: torch.tensor, 
                        token_mask: torch.tensor, start_labels: List[int],
                        gold_entity: torch.tensor, gold_entity_mask: torch.tensor):
        
        context_mask = context_mask.float()
        h = self.bert(input_ids=encodings, attention_mask=context_mask)[0] + 1

        batch_size = encodings.shape[0]
        all_entity_scores = []
        all_entity_preds = []
        all_rel_logits = []

        for batch in range(batch_size): # every batch

            beam_entity = BeamSearch(self._beam_size)
            beam_size = beam_entity.get_beam_size

            num_steps = gold_entity[batch].shape[-1]
            word_h = h[batch].repeat(token_mask[batch].shape[0], 1, 1) * token_mask[batch].unsqueeze(-1)
            word_h_pooled = word_h.max(dim=1)[0]
            word_h_pooled = word_h_pooled[:num_steps+2].contiguous()
            # curr word repr.
            curr_word_reprs = word_h_pooled[1:-1].contiguous().unsqueeze(0).repeat(beam_size, 1, 1)         

            entity_masks = torch.zeros((num_steps + 2, num_steps + 2), dtype = torch.bool).fill_diagonal_(1).to(self._device)
            entity_masks = entity_masks.unsqueeze(0).repeat(beam_size, 1, 1)
            entity_masks[:, 0] = 0

            prev_label = torch.zeros(beam_size, dtype=torch.long).to(self._device)

           # Entity classification.
            for i in range(num_steps): # no [CLS], no [SEP] 

                # curr word repr.
                # print("i:", i)
                curr_word_repr = curr_word_reprs[:, i]
                # mask from previous entity token until current position.

                prev_mask = entity_masks[:, i, :]
                # prvious label embedding.
                # print("prev label:", prev_label)
                prev_label_embedding = self.entity_label_embedding(prev_label)
                # print("prev label embedding:", prev_label_embedding)
                
                prev_entity = word_h_pooled.unsqueeze(0) * prev_mask.unsqueeze(-1)
                prev_entity_pooled = prev_entity.max(dim=1)[0]

                curr_entity_repr = torch.cat([curr_word_repr - 1, prev_entity_pooled - 1], dim=1).unsqueeze(0)
#                 curr_entity_repr = curr_word_repr.unsqueeze(0) - 1
                curr_entity_logits = self.entity_classifier(curr_entity_repr)
                beam_entity.advance(curr_entity_logits)

                curr_label = beam_entity.get_curr_state

                prev_label = curr_label

                istart =  (curr_label % 4 == 1) | (curr_label % 4 == 2) | (curr_label == 0)

                entity_masks = entity_masks[beam_entity.get_curr_origin]
                
                entity_masks[:, i+1] +=  (~istart).unsqueeze(-1) * prev_mask

                entity_masks[:, i, i+1] += (~istart)

                # print("="*50)

            entity_scores, entity_preds = beam_entity.get_best_path
            entity_scores = entity_scores.to(self._device)
            entity_preds = entity_preds.to(self._device)
            all_entity_scores.append(entity_scores)
            all_entity_preds.append(entity_preds)

            # Relation classification.
            curr_rel_logits = self._forward_relation(curr_word_reprs[0], entity_preds, entity_masks[0], gold_entity[batch], True)
            all_rel_logits.append(curr_rel_logits)


        return all_entity_scores, all_entity_preds, all_rel_logits


    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)



class bert_ner(BertPreTrainedModel):
    """ Span-based model to jointly extract entities and relations """

    def __init__(self, config: BertConfig, tokenizer: BertTokenizer,
                 relation_labels: int, entity_labels: int,
                 entity_label_embedding: int,  prop_drop: float, 
                 freeze_transformer: bool, device):
        
        super(bert_ner, self).__init__(config)

        # BERT model
        self.bert = BertModel(config)
        self._tokenizer = tokenizer
        # layers
        self.entity_classifier = nn.Linear(config.hidden_size, entity_labels)
        self.dropout = nn.Dropout(prop_drop)

        self._relation_labels = relation_labels
        self._entity_labels = entity_labels


        # weight initialization
        self.init_weights()

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False


    def _forward_train(self, encodings: torch.tensor, context_mask: torch.tensor): 

        h = self.bert(input_ids=encodings, attention_mask=context_mask)[0]


        h = self.dropout(h)

        all_entity_logits = self.entity_classifier(h)

        # for batch in range(batch_size):

        #     # get cls repr.
        #     cls_id = self._tokenizer.convert_tokens_to_ids("[CLS]")
        #     cls_repr = get_token(h[batch], encodings[batch], cls_id)
            
        #     # align bert token embeddings to word embeddings            
        #     word_h = align_bert_embeddings(h[batch], token_mask[batch], cls_repr)
        #     # print(word_h.shape)

        #     word_h = self.dropout(word_h)

        #     logits = self.entity_classifier(word_h)
            
        #     all_entity_logits.append(logits)

        return all_entity_logits, []


    def _forward_eval(self, encodings: torch.tensor, context_mask: torch.tensor):

        h = self.bert(input_ids=encodings, attention_mask=context_mask)[0]

        h = self.dropout(h)

        all_entity_logits = self.entity_classifier(h)

        # for batch in range(batch_size):

        #     # get cls repr.
        #     cls_id = self._tokenizer.convert_tokens_to_ids("[CLS]")
        #     cls_repr = get_token(h[batch], encodings[batch], cls_id)
            
        #     # align bert token embeddings to word embeddings            
        #     word_h = align_bert_embeddings(h[batch], token_mask[batch], cls_repr)
        #     # print(word_h.shape)

        #     logits = self.entity_classifier(word_h)

        #     all_entity_logits.append(logits)

        return all_entity_logits, []

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)


# Model access

_MODELS = {
    'table_filling': TableF,
    'bert_ner': bert_ner,
}


def get_model(name):
    return _MODELS[name]
