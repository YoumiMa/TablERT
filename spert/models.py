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

from typing import List

import torchcrf
import json

def get_token(h: torch.tensor, token_mask: torch.tensor):
    """ Get specific token embedding masked by token_mask. """

    emb_size = h.shape[-1]
    batch_size = h.shape[0]

    token_mask = token_mask.unsqueeze(2).repeat(1, 1, emb_size)
    token_h = torch.masked_select(h, token_mask).view(batch_size, -1, emb_size)
    # get contextualized embedding of given token (maxpooling)
    print("picked:", token_mask)
    if token_h.nelement() != 0:
        token_h = token_h.max(dim=1)[0]

    return token_h


class SpERT(BertPreTrainedModel):
    """ Span-based model to jointly extract entities and relations """

    def __init__(self, config: BertConfig, tokenizer: BertTokenizer,
                 relation_labels: int, entity_labels: int,
                 entity_label_embedding:int,  prop_drop: float, 
                 freeze_transformer: bool, max_pairs: int = 100):
        super(SpERT, self).__init__(config)

        # BERT model
        self.bert = BertModel(config)
        self._tokenizer = tokenizer
        # layers
        self.entity_label_embedding = nn.Embedding(entity_labels , entity_label_embedding)
        self.rel_classifier = nn.Linear(config.hidden_size * 5, relation_labels)
        self.entity_classifier = nn.Linear(config.hidden_size , entity_labels)
        self.crf = torchcrf.CRF(entity_labels)
        self.dropout = nn.Dropout(prop_drop)

        self._relation_labels = relation_labels
        self._entity_labels = entity_labels
        self._max_pairs = max_pairs

        # weight initialization
        self.init_weights()

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def _forward_train(self, encodings: torch.tensor, context_mask: torch.tensor, 
                        entity_gt: torch.tensor, tables: torch.tensor, 
                        entity_boundary: torch.tensor, curr_index: int, token_mask: torch.tensor):
        
        # get contextualized token embeddings from last transformer layer
        context_mask = context_mask.float()
        h = self.bert(input_ids=encodings, attention_mask=context_mask)[0]

        batch_size = encodings.shape[0]
        context_size = encodings.shape[1]

        # print(ids)
        # representation of current encoding.
        curr_repr = get_token(h, token_mask[:, curr_index])


        # pick out from previous entity to curr_index.
        mask = torch.zeros_like(encodings, dtype=torch.float)

        for i in range(batch_size):
            prev = 0
            boundary_mask = entity_boundary[i,:curr_index]
            if boundary_mask.nelement() != 0:
                x = boundary_mask < entity_boundary[i,curr_index-1]
                # print(x)
                nonzeros = x.nonzero().squeeze()
                # print(nonzeros)
                if nonzeros.nelement() != 0:
                    prev = nonzeros.max()
            mask[i, prev+1: curr_index] = 1


        # print("current:", curr_index)
        # print("gold tag seq:", entity_gt)
        # print("entity boundary:", entity_boundary)
        # print(mask)


        # maxpool between previous entity and current position.
        masked_repr = mask.unsqueeze(2) * h
        masked_repr_pool = masked_repr.max(dim=1)[0]

        # previous labels.
        prev_labels = self.entity_label_embedding(entity_gt)
        prev_label = torch.index_select(prev_labels, 1, entity_boundary[:,prev].clone()).view(batch_size, -1)
        # print("labels:", prev_labels)
        # print("previous label:", prev, prev_label)
        # print("=============")
        # average pool between label embeddings.
        prev_label_pool = (prev_labels * mask.unsqueeze(2)).sum(dim=1)
        if mask.sum() != 0:
            prev_label_pool /= mask.sum().item()



        # entity_repr = torch.cat([curr_repr, masked_repr_pool, prev_label, prev_label_pool], dim=1)
        entity_repr = curr_repr

        entity_repr = self.dropout(entity_repr)

        entity_logits = self.entity_classifier(entity_repr)

        if entity_logits.nelement() != 0:
            tables[:, curr_index] = entity_logits
        
        return entity_logits, tables

    def _forward_eval(self, encodings: torch.tensor, context_mask: torch.tensor, entity_masks: torch.tensor,
                      entity_sizes: torch.tensor, tokens: List[Token], entity_spans: torch.tensor = None,
                      entity_sample_mask: torch.tensor = None):
        # get contextualized token embeddings from last transformer layer
        context_mask = context_mask.float()
        h = self.bert(input_ids=encodings, attention_mask=context_mask)[0]

        entity_masks = entity_masks.float()
        batch_size = encodings.shape[0]
        ctx_size = context_mask.shape[-1]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks)


        ### softmax ###
        # # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        # relations, rel_masks, rel_sample_masks = self._filter_spans(entity_clf, entity_spans,entity_sample_mask, ctx_size)
        ### softmax ###


        ### CRF ###
        entity_path = self.crf._viterbi_decode(entity_clf, torch.ones_like(encodings, dtype=torch.uint8))
        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        entity_path = torch.tensor(entity_path)
        relations, rel_masks, rel_sample_masks = self._filter_spans_crf(entity_path, entity_spans,
                                                                    entity_sample_mask, ctx_size)
        # ### CRF ###
        rel_masks = rel_masks.float()
        rel_sample_masks = rel_sample_masks.float()
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(encodings, entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            # apply sigmoid
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        rel_clf = rel_clf * rel_sample_masks  # mask
        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, entity_path, rel_clf, relations

    def _classify_relations(self, encodings, entity_spans, size_embeddings, relations, rel_masks, h, chunk_start):
        batch_size = relations.shape[0]

        # create chunks if necessary
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :relations.shape[1], :]

        # print(relations)
        # get pairs of entity candidate representations
        entity_pairs = util.batch_index(entity_spans, relations)
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        # get corresponding size embeddings
        size_pair_embeddings = util.batch_index(size_embeddings, relations)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)
        # relation context (context in-between an entity candidate pair)
        rel_ctx = rel_masks * h
        rel_ctx = rel_ctx.max(dim=2)[0]

        # create relation candidate representations including context, max pooled entity candidate pairs
        # and corresponding size embeddings
        rel_repr = torch.cat([rel_ctx, entity_pairs, size_pair_embeddings], dim=2)
        rel_repr = self.dropout(rel_repr)

        # classify relation candidates
        chunk_rel_logits = self.rel_classifier(rel_repr)
        return chunk_rel_logits


    def _filter_spans_crf(self, entity_clf, entity_spans, entity_sample_mask, ctx_size):
        batch_size = 1
        entity_logits_max = entity_clf
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            # get spans classified as entities
            non_zero_indices, non_zero_spans = self._get_indices_and_spans(entity_logits_max, entity_spans[i])
            # create relations and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.rel_classifier.weight.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device).unsqueeze(-1)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device).unsqueeze(-1)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_mask, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1)  # get entity type (including none)
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            # get spans classified as entities
            non_zero_indices, non_zero_spans = self._get_indices_and_spans(entity_logits_max[i], entity_spans[i])
            # create relations and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.rel_classifier.weight.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device).unsqueeze(-1)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device).unsqueeze(-1)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks



    def _get_indices_and_spans(self, entity_logits_max, all_spans):

        non_zero_indices = (entity_logits_max != 0).nonzero().view(-1)
        non_zero_spans = []
        s = 0
        for i in range(entity_logits_max.shape[0]-1):
            if i in non_zero_indices and i+1 not in non_zero_indices:
                non_zero_spans.append([non_zero_indices[s].item(), i+1])
                while  s <  non_zero_indices.shape[0] and non_zero_indices[s].item() < i+1:
                    s += 1

        indices = []
        for span in non_zero_spans:
            for j in range(all_spans.shape[0]):
                if all_spans[j][0].item() ==  span[0] and all_spans[j][1].item() ==  span[1]:
                    indices.append(j)
                    break

        return indices, non_zero_spans

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)


# Model access

_MODELS = {
    'spert': SpERT,
}


def get_model(name):
    return _MODELS[name]
