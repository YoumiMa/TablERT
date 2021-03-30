import torch
from torch import nn as nn
from transformers import BertConfig, AlbertConfig
from transformers import BertModel, AlbertModel
from transformers import BertPreTrainedModel, AlbertPreTrainedModel
from transformers import BertTokenizer, AlbertTokenizer


from table_filling import sampling
from table_filling import util


from table_filling.entities import Token
from table_filling.attention import MultiHeadAttention
from table_filling.beam import Beam


from typing import List


class TableF(BertPreTrainedModel):
    
    """ The table filling model for jointly extracting entities and relations using pretrained BERT.
    
    Params:
    :config: configuration for pretrained BERT;
    :tokenizer: pretraind BERT tokenizer;
    :relation_labels: number of relation labels;
    :entity_labels: number of entity labels;
    :entity_label_embedding: dimension of NE label embedding;
    :att_hidden: dimension of hidden attention for relation classification;
    :prop_drop: dropout rate;
    :freeze_transformer: fix transformer parameters or not;
    :device: devices to run the model at, e.g. "cuda:1" or "cpu".
    
    """

    def __init__(self, config: BertConfig, tokenizer: BertTokenizer,
                 entity_labels: int, relation_labels: int, 
                 entity_label_embedding: int,  att_hidden: int,
                 prop_drop: float, beam_size: int, 
                 freeze_transformer: bool, device):
        super(TableF, self).__init__(config)

        # BERT model
        self.bert = BertModel(config)
        self._tokenizer = tokenizer
        
        self._device = device
        
        # task-specific layers for NER
        self.entity_label_embedding = nn.Embedding(entity_labels , entity_label_embedding)
        self.entity_classifier = nn.Linear(config.hidden_size * 2 + entity_label_embedding, entity_labels)
        
        # task-specific layers for RE
        self.rel_classifier = MultiHeadAttention(relation_labels, config.hidden_size + entity_label_embedding , att_hidden , device)
        

        self._beam_size = beam_size
        self.dropout = nn.Dropout(prop_drop)
        
        self._entity_labels = entity_labels
        self._relation_labels = relation_labels
        
        # weight initialization
        self.init_weights()

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

                
    def _update_mask(self, i: int, labels: torch.tensor, mask: torch.tensor):
        
        """ Function for updating entity span masks given NE label predictions until current timestep.
        
        Params:
        :i: current timestep;
        :labels: label predictions until timestep i, a tensor of shape (beam_size, i+1);
        :mask: entity span masks, a tensor of shape (beam_size, n+2, n+2), n=sequence length, +2 since +[CLS]+[SEP].
                dim1 of mask =  timestep, dim2 of mask = entity span mask known at each timestep.
                
        Return:
        :mask: updated entity span masks of shape (beam_size, n+2, n+2) a t timestep i.
        
        """
        
        is_O = labels == 0
        is_B = labels % 4 == 1
        is_U = labels % 4 == 2
        is_I = labels % 4 == 3
        is_L = (labels % 4 == 0) & (~is_O)

        # labels started with B and I need to be considered as a continious span
        is_BI = is_B + is_I
        start_p = is_B + is_U + is_O
        
        # update entity span mask at position i
        # record every start position increasement
        boundary =  ( start_p.cumsum(dim=1) - (start_p.cumsum(dim=1).max(dim=1)[0] - 1).view(self._beam_size, 1) ) > 0
        # connect the start boundary with previous B-X or I-X to make a contineous span
        # start from 1 since 0=[CLS]
        mask[:, i, 1:i+1] += is_BI & boundary 
        prev_mask = mask[:, i, :]       
        
        # update the whole entity span mask 
        # only record those newly included span info at visited positions(i.e., position <= i).
        temp = prev_mask.unsqueeze(-1).repeat(1, 1, mask.shape[-1])
        span_mask = temp & temp.transpose(2,1)
        mask[span_mask] = True
        
        return mask

    def _forward_token(self, h: torch.tensor, token_mask: torch.tensor, 
                           gold_seq: torch.tensor, entity_mask: torch.tensor):

        ''' Forward step for predicting NEs.
        
        Params:
        :h: subword representations from pretrained BERT, of shape (1, subword_seq_length, BERT_dim);
        :token_mask: a tensor mapping subword to word (token), of shape (1, n+2, subword_seq_length);
        :gold_seq: ground-truth sequence for NE labels, of shape (1, n+2);
        :entity_mask: ground-truth mask for NE spans, of shape (1, n+2, n+2).
        
        Return:
        :curr_word_repr: word representations from pooling of sub-word representations, of shape (1, n, BERT_dim);
        :curr_entity_logits: NE scores for each word, of shape (1, n, entity_labels).
        
        '''
        
        num_steps = gold_seq.shape[-1]
        
        # map from subword repr to word repr.
        word_h = h.repeat(token_mask.shape[0], 1, 1) * token_mask.unsqueeze(-1)
        word_h_pooled = word_h.max(dim=1)[0]
        word_h_pooled = word_h_pooled[:num_steps+2].contiguous()
        word_h_pooled[0,:] = 0

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

        # entity repr for classification.
        entity_repr = torch.cat([curr_word_repr - 1, prev_entity_pooled - 1, prev_label], dim=1).unsqueeze(0)
        entity_repr = self.dropout(entity_repr)
        
        curr_entity_logits = self.entity_classifier(entity_repr)

        return curr_word_repr, curr_entity_logits


    def _forward_relation(self, h: torch.tensor,  entity_preds: torch.tensor, 
                          entity_mask: torch.tensor):

        ''' Forward step for predicting relations.
        
        Params:
        :h: word representations from pooling of sub-word representations, of shape (batch_size, n, BERT_dim);
        :entity_preds: predicted sequences of NEs, of shape (batch_size, n);
        :entity_mask: predicted span masks of NEs, of shape (batch_size, n, n).
        
        Return:
        :rel_logits: relation scores for each word pair, of shape (1, relation_labels, n, n).
        
        '''
        entity_labels = entity_preds

        # entity span repr.
        masks_no_cls_rep = entity_mask[:, 1:-1, 1:-1]
        entity_repr = h.repeat(masks_no_cls_rep.shape[-1], 1, 1) * masks_no_cls_rep.unsqueeze(-1)
        entity_repr_pool = entity_repr.max(dim=1)[0]

        # entity_label span repr.
        entity_label_embeddings = self.entity_label_embedding(entity_labels)     
        entity_label_repr = entity_label_embeddings.unsqueeze(1) * masks_no_cls_rep.unsqueeze(-1)
        entity_label_pool = entity_label_repr.max(dim=2)[0]
        
        # rel repr for classification.
        rel_embedding = torch.cat([entity_repr_pool - 1, entity_label_pool], dim=2)    
        rel_embedding = self.dropout(rel_embedding)

        rel_logits = self.rel_classifier(rel_embedding, rel_embedding, rel_embedding)            
        return rel_logits


    def _forward_train(self, encodings: torch.tensor, context_mask: torch.tensor, 
                        token_mask: torch.tensor, gold_entity: torch.tensor, entity_masks: List[torch.tensor], gold_rel:torch.tensor, allow_rel: bool):  
        
        ''' Forward step for training.
        
        Params:
        :encodings: token encodings (in subword), of shape (batch_size, subword_sequence_length);
        :context_mask: masking out [PAD] from encodings, of shape (batch_size, subword_squence_length);
        :token_mask: a tensor mapping subword to word (token), of shape (batch_size, n+2, subword_sequence_length);
        :gold_entity: ground-truth sequence for NE labels, of shape (batch_size, n+2);
        :entity_masks: ground-truth mask for NE spans, of shape (batch_size, n+2, n+2);
        :gold_rel: ground-truth matrices for relation labels, of shape (batch_size, n+2, n+2);        
        :allow_rel: whether allow re predictions or not.
        
        Return:
        
        :all_entity_logits: NE scores for each word on each batch, a list of length=batch_size containing tensors of shape (1, n, entity_labels);
        :all_rel_logits: relation scores for each word pair on each batch, a list of length=batch_size containing tensors of shape (1, relation_labels, n, n).
        
        '''
        
        # get contextualized token embeddings from last transformer layer
        context_mask = context_mask.float()
        h = self.bert(input_ids=encodings, attention_mask=context_mask)[0] + 1

        batch_size = encodings.shape[0]
        all_entity_logits = []
        all_rel_logits = []

        for batch in range(batch_size): # every batch
            
            entity_mask = entity_masks[batch]
            
            # NE forward steps.
            curr_word_reprs, curr_entity_logits = self._forward_token(h[batch], token_mask[batch], gold_entity[batch], entity_mask)
            all_entity_logits.append(curr_entity_logits)
            entity_preds = torch.argmax(curr_entity_logits, dim=2)
            
            # For RE, Use diagonal entity masks for training.
            diag_entity_mask = torch.zeros_like(entity_mask, dtype=torch.bool).to(self._device).fill_diagonal_(1)
            num_steps = gold_entity[batch].shape[-1]            
                
            # unsqueeze the first dimension to match (beam_size, n+2, n+2) for evaluation
            curr_rel_logits = self._forward_relation(curr_word_reprs, entity_preds, diag_entity_mask.unsqueeze(0))
            all_rel_logits.append(curr_rel_logits)

        if allow_rel:
            return all_entity_logits, all_rel_logits
        else:
            return all_entity_logits, []

    
    def _forward_eval(self, encodings: torch.tensor, context_mask: torch.tensor, token_mask: torch.tensor):
    
        ''' Forward step for evaluation.
        
        Params:
        :encodings: token encodings (in subword), of shape (batch_size, subword_sequence_length);
        :context_mask: masking out [PAD] from encodings, of shape (batch_size, subword_squence_length);
        :token_mask: a tensor mapping subword to word (token), of shape (batch_size, n+2, subword_sequence_length).
        
        Return:
        :all_entity_scores: NE scores for each beam on each batch, a list of length=batch_size containing tensors of shape (beam_size,);
        :all_entity_preds: NE predictions for each word on each batch, a list of length=batch_size containing tensors of shape (beam_size, n, entity_labels);
        :all_rel_logits: relation scores for each word pair on each batch, a list of length=batch_size containing tensors of shape (1, relation_labels, n, n).        
        
        '''
        
        context_mask = context_mask.float()
        h = self.bert(input_ids=encodings, attention_mask=context_mask)[0] + 1

        batch_size = encodings.shape[0]
        all_entity_scores = []
        all_entity_preds = []
        all_rel_logits = []

        for batch in range(batch_size): # every batch


            num_steps = token_mask[batch].sum(axis=1).nonzero().shape[0] - 2

            word_h = h[batch].repeat(token_mask[batch].shape[0], 1, 1) * token_mask[batch].unsqueeze(-1)
            word_h_pooled = word_h.max(dim=1)[0]
            word_h_pooled = word_h_pooled[:num_steps+2].contiguous()
            word_h_pooled[0,:] = 0

            # curr word repr.
            curr_word_reprs = word_h_pooled[1:-1].contiguous()

            entity_masks = torch.zeros((num_steps + 2, num_steps + 2), dtype = torch.bool).fill_diagonal_(1).to(self._device)
            entity_masks = entity_masks.unsqueeze(0).repeat(self._beam_size, 1, 1)
            # beam search.
            beam = Beam(self._beam_size)

           # Entity classification.
            for i in range(num_steps): # no [CLS], no [SEP] 
                # curr word repr.
                curr_word_repr = curr_word_reprs[i].unsqueeze(0).repeat(self._beam_size, 1)
                
                # mask from previous entity token until current position.
                if i == 0:
                    prev_labels = torch.zeros((self._beam_size, 1), dtype=torch.long, device=self._device)
                
                prev_label_repr = self.entity_label_embedding(prev_labels[:, -1])
                
                prev_masks = entity_masks[:, i, :]
                
                prev_entity = word_h_pooled.repeat(prev_masks.shape[0], 1, 1) * prev_masks.unsqueeze(-1)
                prev_entity_pooled = prev_entity.max(dim=1)[0] 
                
                curr_entity_repr = torch.cat([curr_word_repr - 1, prev_entity_pooled - 1, prev_label_repr], dim=1)
                curr_entity_logits = self.entity_classifier(curr_entity_repr)
                beam.advance(curr_entity_logits)
                
                # get best k candidates and update the entity span mask.
                prev_labels = torch.stack([torch.stack(beam.get_hyp(b), dim=0) for b in range(self._beam_size)], dim=0)
                entity_masks = self._update_mask(i+1, prev_labels, entity_masks)
           
            entity_scores = beam.get_score
            entity_preds = prev_labels
            
            # Relation classification.
            curr_rel_logits = self._forward_relation(curr_word_reprs, entity_preds, entity_masks)
            all_entity_scores.append(entity_scores)
            all_entity_preds.append(entity_preds)
            all_rel_logits.append(curr_rel_logits)


        return all_entity_scores, all_entity_preds, all_rel_logits


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
