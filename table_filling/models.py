import torch
from torch import nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel
from transformers import BertTokenizer


from table_filling import sampling
from table_filling import util

from table_filling.entities import Token
from table_filling.loss import TableLoss, Loss
from table_filling.attention import MultiHeadAttention


from typing import List


class TableF(BertPreTrainedModel):
    """ table filling model to jointly extract entities and relations """

    def __init__(self, config: BertConfig, tokenizer: BertTokenizer,
                 relation_labels: int, entity_labels: int,
                 entity_label_embedding: int,  att_hidden: int,
                 prop_drop: float, freeze_transformer: bool, device):
        super(TableF, self).__init__(config)

        # BERT model
        self.bert = BertModel(config)
        self._tokenizer = tokenizer
        self._device = device
        # layers
        self.entity_label_embedding = nn.Embedding(entity_labels , entity_label_embedding)
        self.entity_classifier = nn.Linear(config.hidden_size * 2 + entity_label_embedding, entity_labels)
        self.rel_classifier = MultiHeadAttention(relation_labels, config.hidden_size + entity_label_embedding, att_hidden , device)
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


    def _forward_token(self, h: torch.tensor, token_mask: torch.tensor, 
                           gold_seq: torch.tensor, entity_mask: torch.tensor):

        num_steps = gold_seq.shape[-1]
        word_h = h.repeat(token_mask.shape[0], 1, 1) * token_mask.unsqueeze(-1)
        word_h_pooled = word_h.max(dim=1)[0]
        word_h_pooled = word_h_pooled[:num_steps+2].contiguous()

        # curr word repr.
        curr_word_repr = word_h_pooled[1:-1].contiguous()

        # prev entity repr.
        prev_entity = torch.tril(entity_mask, diagonal=0)

        prev_entity_h = word_h_pooled.repeat(prev_entity.shape[0], 1, 1) * prev_entity.unsqueeze(-1)
        prev_entity_pooled = prev_entity_h.max(dim=1)[0]
        prev_entity_pooled = prev_entity_pooled[:num_steps].contiguous()

        # prev_label_embedding.
        prev_seq = torch.cat([torch.tensor([0]).to(self._device), gold_seq])
        prev_label = self.entity_label_embedding(prev_seq[:-1])

        entity_repr = torch.cat([self.dropout(curr_word_repr) - 1, self.dropout(prev_entity_pooled) - 1, prev_label], dim=1).unsqueeze(0)

        curr_entity_logits = self.entity_classifier(entity_repr)

        return curr_word_repr, curr_entity_logits


    def _forward_relation(self, h: torch.tensor,  entity_preds: torch.tensor, 
                          entity_mask: torch.tensor, is_eval: bool = False):


        entity_labels = entity_preds.unsqueeze(0)

        # entity repr.
        masks_no_cls_rep = entity_mask[1:-1, 1:-1]
        entity_repr = h.repeat(masks_no_cls_rep.shape[-1], 1, 1) * masks_no_cls_rep.unsqueeze(-1)
        entity_repr_pool = entity_repr.max(dim=1)[0]

        # entity_label repr.
        entity_label_embeddings = self.entity_label_embedding(entity_labels)
        entity_label_repr = entity_label_embeddings.repeat(masks_no_cls_rep.shape[-1], 1, 1) * masks_no_cls_rep.unsqueeze(-1)
        entity_label_pool = entity_label_repr.max(dim=1)[0]


        rel_embedding = torch.cat([self.dropout(entity_repr_pool).unsqueeze(0) - 1, entity_label_pool.unsqueeze(0)], dim=2)
        rel_logits = self.rel_classifier(rel_embedding, rel_embedding, rel_embedding)

        return rel_logits

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

        for batch in range(batch_size): # every batch
            

            entity_mask = entity_masks[batch]

            word_h, curr_entity_logits = self._forward_token(h[batch], token_mask[batch], gold_entity[batch], entity_mask)

            entity_preds = torch.argmax(curr_entity_logits, dim=2)
            
            diag_entity_mask = torch.zeros_like(entity_mask, dtype=torch.bool).to(self._device).fill_diagonal_(1)

            all_entity_logits.append(curr_entity_logits)
            # Relation classification.

            num_steps = gold_entity[batch].shape[-1]
            word_h = h[batch].repeat(token_mask[batch].shape[0], 1, 1) * token_mask[batch].unsqueeze(-1)
            word_h_pooled = word_h.max(dim=1)[0]
            word_h_pooled = word_h_pooled[:num_steps+2].contiguous()

            # curr word repr.
            curr_word_repr = word_h_pooled[1:-1].contiguous()
            curr_rel_logits = self._forward_relation(curr_word_repr, gold_entity[batch] , diag_entity_mask)
            all_rel_logits.append(curr_rel_logits)

        if allow_rel:
            return all_entity_logits, all_rel_logits
        else:
            return all_entity_logits, []

    
    def _forward_eval(self, encodings: torch.tensor, context_mask: torch.tensor, token_mask: torch.tensor, gold_entity: torch.tensor):
                
        context_mask = context_mask.float()
        h = self.bert(input_ids=encodings, attention_mask=context_mask)[0] + 1

        batch_size = encodings.shape[0]
        all_entity_logits = []
        all_entity_scores = []
        all_entity_preds = []
        all_rel_logits = []

        for batch in range(batch_size): # every batch


            num_steps = token_mask[batch].sum(axis=1).nonzero().shape[0] - 2


            word_h = h[batch].repeat(token_mask[batch].shape[0], 1, 1) * token_mask[batch].unsqueeze(-1)
            word_h_pooled = word_h.max(dim=1)[0]
            word_h_pooled = word_h_pooled[:num_steps+2].contiguous()
            # curr word repr.
            curr_word_reprs = word_h_pooled[1:-1].contiguous()

            entity_masks = torch.zeros((num_steps + 2, num_steps + 2), dtype = torch.bool).fill_diagonal_(1).to(self._device)

            entity_preds = torch.zeros((num_steps + 1, 1), dtype=torch.long).to(self._device)
            entity_logits = []
            entity_scores = torch.zeros((num_steps, 1), dtype=torch.float).to(self._device)

#            # Entity classification.
#             for i in range(num_steps): # no [CLS], no [SEP] 

#                 # curr word repr.
#                 curr_word_repr = curr_word_reprs[i].unsqueeze(0)
#                 # mask from previous entity token until current position.

#                 prev_mask = entity_masks[i, :]
     
#                 prev_label_repr = self.entity_label_embedding(entity_preds[i])
                
#                 prev_entity = word_h_pooled.unsqueeze(0) * prev_mask.unsqueeze(-1)
#                 prev_entity_pooled = prev_entity.max(dim=1)[0]

#                 curr_entity_repr = torch.cat([curr_word_repr - 1, prev_entity_pooled - 1, prev_label_repr], dim=1).unsqueeze(0)
#                 curr_entity_logits = self.entity_classifier(curr_entity_repr)
#                 entity_logits.append(curr_entity_logits.squeeze(1))

#                 curr_label = curr_entity_logits.argmax(dim=2).squeeze(0)
#                 entity_scores[i] += torch.softmax(curr_entity_logits, dim=2).max(dim=2)[0].squeeze(0)
#                 entity_preds[i+1] = curr_label

#                 istart =  (curr_label % 4 == 1) | (curr_label % 4 == 2) | (curr_label == 0)
                
#                 # update entity mask for the next time step            
#                 entity_masks[i+1] +=  (~istart) * prev_mask

#                 # update entity span info for all time-steps
#                 entity_masks[prev_mask.nonzero()[0].item():i+1, i+1] += (~istart).squeeze(0)

#             all_entity_logits.append(torch.stack(entity_logits, dim=1))
            all_entity_scores.append(torch.t(entity_scores.squeeze(-1)))
#             all_entity_preds.append(torch.t(entity_preds[1:].squeeze(-1)))

            # Relation classification.
#             curr_rel_logits = self._forward_relation(curr_word_reprs, entity_preds[1:].squeeze(-1), entity_masks, True)

            curr_rel_logits = self._forward_relation(curr_word_reprs, gold_entity[batch], entity_masks, True)
            all_rel_logits.append(curr_rel_logits)


        return all_entity_logits, all_entity_scores, all_entity_preds, all_rel_logits


    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)



# Model access

_MODELS = {
    'table_filling': TableF,
}


def get_model(name):
    return _MODELS[name]