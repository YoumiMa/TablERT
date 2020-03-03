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

from typing import List

# import torchcrf
import json
import itertools

def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h


def align_bert_embeddings(h: torch.tensor, token_mask: torch.tensor, cls_repr: torch.tensor):
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
                 relation_labels: int, entity_labels: int,
                 entity_label_embedding: int,  prop_drop: float, 
                 freeze_transformer: bool, device):
        super(TableF, self).__init__(config)

        # BERT model
        self.bert = BertModel(config)
        self._tokenizer = tokenizer
        # layers
        self.entity_label_embedding = nn.Embedding(entity_labels , entity_label_embedding)
        self.entity_classifier = nn.Linear(config.hidden_size * 2 + entity_label_embedding , entity_labels)
        # sel.crf = torchcrf.CRF(entity_labels)
        self.dropout = nn.Dropout(prop_drop)
        self.attn = MultiHeadAttention(relation_labels, config.hidden_size + entity_label_embedding , 5 , device)

        self._relation_labels = relation_labels
        self._entity_labels = entity_labels


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
                prev_mask: torch.tensor, prev_embedding: torch.tensor,
                is_eval: bool):

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
        if not is_eval:
            # print(entity_repr)
            entity_repr = self.dropout(entity_repr)
            # print("dropeed:", entity_repr)

        entity_logits = self.entity_classifier(entity_repr)

        return entity_logits

    def _forward_relation(self, h: torch.tensor, entity_logits: torch.tensor):

        entity_labels = torch.argmax(entity_logits, dim=2)
        # entity_labels = gold_entity[batch].unsqueeze(0)
        entity_label_embeddings = self.entity_label_embedding(entity_labels)
        rel_embedding = torch.cat([h[1:-1,:].unsqueeze(0).contiguous(), entity_label_embeddings], dim=2)
        rel_embedding = self.dropout(rel_embedding)
        att, _ = self.attn(rel_embedding, rel_embedding, rel_embedding)
        
        return att.permute(0,2,3,1).contiguous()

    def _forward_relation_old(self, h: torch.tensor, token_mask: torch.tensor, 
                i: int,  j: int, i_embedding: torch.tensor, j_embedding: torch.tensor,
                entity_masks: torch.tensor, is_eval: bool):


        # print(entity_masks)
        context_size = entity_masks.shape[-1]
        i_repr = h[i].unsqueeze(0)
        j_repr = h[j].unsqueeze(0)
        
        # maxpool between previous entity and current position.
        before_mask = entity_masks[i].contiguous()
        before_mask = ~before_mask
        before_mask[i:] = 0 
        before_mask[0] = 1
        # print("before mask:", before_mask)
        before_repr = h[:context_size][before_mask]
        before_pool = before_repr.max(dim=0)[0].unsqueeze(0)

        between_mask = ~entity_masks[i].contiguous() & ~entity_masks[j].contiguous()
        between_mask[:i] = 0
        between_mask[j:] = 0
        # print("between mask:", between_mask)
        between_repr = h[:context_size][between_mask]
        if between_repr.nelement() != 0:
            between_pool = between_repr.max(dim=0)[0].unsqueeze(0)
        else:
            between_pool = torch.zeros_like(before_pool)

        after_mask = entity_masks[j].contiguous()
        after_mask = ~after_mask
        after_mask[:j] = 0
        after_mask[-1] = 1
        # print("after mask:", after_mask)
        after_repr = h[:context_size][after_mask]
        # print(after_repr.shape)
        after_pool = after_repr.max(dim=0)[0].unsqueeze(0)

        # print("=" * 50)
        # concat them for linear classification.
        rel_repr = torch.cat([before_pool, i_repr, between_pool, j_repr, after_pool, i_embedding, j_embedding], dim=1)
        # print(rel_repr.shape)
        # dropout.

        if not is_eval:
            rel_repr = self.dropout(rel_repr)

        rel_logits = self.rel_classifier(rel_repr)

        return rel_logits

    def _forward_train(self, encodings: torch.tensor, context_mask: torch.tensor, 
                        token_mask: torch.tensor, start_labels: List[int], gold_entity: torch.tensor,
                        allow_rel: bool):  
        # get contextualized token embeddings from last transformer layer
        context_mask = context_mask.float()
        h = self.bert(input_ids=encodings, attention_mask=context_mask)[0]

        batch_size = encodings.shape[0]
        # print("encodings:", encodings)
        # print(context_mask * encodings, (context_mask * encodings).shape)
        # context_size = encodings.shape[1]


        all_entity_logits = []
        all_rel_logits = []

        for batch in range(batch_size): # every batch
            

            # get cls repr.
            cls_id = self._tokenizer.convert_tokens_to_ids("[CLS]")
            cls_repr = get_token(h[batch], encodings[batch], cls_id)
            
            # align bert token embeddings to word embeddings            
            word_h = align_bert_embeddings(h[batch], token_mask[batch], cls_repr)
            # print(word_h.shape)
            

            # word_h = h[batch]
            # context_size = context_mask[batch].long().sum().item()
            context_size = word_h.shape[0]
            # print("word_h:", word_h.shape)
            entity_masks = torch.zeros((context_size, context_size), dtype=torch.bool).cuda() 

            entity_logits_batch = []
            rel_logits_batch = []
            prev_i = 0
            prev_label = 0

            # word_pairs = itertools.combinations(range(context_size-2), 2)
            
            # Entity classification.
            for i in range(1, context_size-1): # no [CLS], no [SEP] 
                # mask from previous entity token until current position.
                # prev_mask = self._get_prev_mask_(prev_i, i, context_mask[batch])

                # word
                prev_mask = self._get_prev_mask(prev_i, i, context_size)
                # print("mask:", prev_mask)

                # prvious label embedding.
                prev_embedding = self.entity_label_embedding(torch.tensor(prev_label).cuda())
                
                curr_entity_logits = self._forward_token(word_h, i, prev_mask, prev_embedding, False)
                # print("logits:", curr_entity_logits)

                # prediction of current entity. (GREEDY)
                curr_label = torch.argmax(curr_entity_logits)
                # curr_label = gt_entity[batch][i]

                entity_masks[i, i] = 1                
                # update info of previous entity.
                if curr_label == 0 and prev_label == 0:
                    pass
                elif curr_label in start_labels:
                    prev_i = i
                    prev_label = curr_label
                else:
                    entity_masks[prev_i:i+1, prev_i:i+1] = 1     

                entity_logits_batch.append(curr_entity_logits)

            curr_entity_logits = torch.stack(entity_logits_batch, dim=1)
            all_entity_logits.append(curr_entity_logits)
            # Relation classification.
            if allow_rel:

                curr_rel_logits = self._forward_relation(word_h, curr_entity_logits)
                all_rel_logits.append(curr_rel_logits)
                        
        return all_entity_logits, all_rel_logits

    def _forward_eval(self, encodings: torch.tensor, context_mask: torch.tensor, 
                        token_mask: torch.tensor, start_labels: List[int], gold_entity: torch.tensor):
        
        context_mask = context_mask.float()
        h = self.bert(input_ids=encodings, attention_mask=context_mask)[0]

        batch_size = encodings.shape[0]
        # print("encodings:", encodings)
        # print("tokens:", self._tokenizer.convert_ids_to_tokens([e.item() for e in encodings[0]]))
        # print(context_mask * encodings, (context_mask * encodings).shape)
        # context_size = encodings.shape[1]
        # print("token mask:", token_mask)

        all_entity_logits = []
        all_rel_logits = []

        for batch in range(batch_size): # every batch
            

            # get cls repr.
            cls_id = self._tokenizer.convert_tokens_to_ids("[CLS]")
            cls_repr = get_token(h[batch], encodings[batch], cls_id)
            
            # align bert token embeddings to word embeddings            
            word_h = align_bert_embeddings(h[batch], token_mask[batch], cls_repr)
            # word_h = h[batch]
            # context_size = context_mask[batch].long().sum().item()

            context_size = word_h.shape[0]
            entity_masks = torch.zeros((context_size, context_size), dtype=torch.bool).cuda() 

            entity_logits_batch = []
            rel_logits_batch = []
            prev_i = 0
            prev_label = 0

            # Entity classification.
            for i in range(1, context_size-1): # no [CLS], no [SEP] 
                # mask from previous entity token until current position.
                # prev_mask = self._get_prev_mask_(prev_i, i, context_mask[batch])
                prev_mask = self._get_prev_mask(prev_i, i, context_size)
                # print(i, "prev mask:", prev_mask)
                # prvious label embedding.
                prev_embedding = self.entity_label_embedding(torch.tensor(prev_label).cuda())
                
                curr_entity_logits = self._forward_token(word_h, i, prev_mask, prev_embedding, True)
                # print(curr_entity_logits.shape)

                # prediction of current entity. (GREEDY)
                curr_label = torch.argmax(curr_entity_logits)
                # curr_label = gt_entity[batch][i]

                entity_masks[i, i] = 1                
                # update info of previous entity.(VER1: comment below)
                if curr_label == 0 and prev_label == 0:
                    pass
                elif curr_label in start_labels:
                    prev_i = i
                    prev_label = curr_label
                else:
                    entity_masks[prev_i:i+1, prev_i:i+1] = 1   

                entity_logits_batch.append(curr_entity_logits)  
            
            curr_entity_logits = torch.stack(entity_logits_batch, dim=1)
            all_entity_logits.append(curr_entity_logits)

            # print(all_entity_logits)
            # print("entity mask:", entity_masks)
            

            # Relation classification.

            curr_rel_logits = self._forward_relation(word_h, curr_entity_logits)
            all_rel_logits.append(curr_rel_logits)

        # apply softmax
        for batch in range(batch_size):
            # print(all_entity_logits[batch])
            all_entity_logits[batch] = torch.softmax(all_entity_logits[batch], dim=2)
            # print("after softmax:", all_entity_logits[batch])
            all_rel_logits[batch] = torch.softmax(all_rel_logits[batch], dim=3)

        return all_entity_logits, all_rel_logits


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
