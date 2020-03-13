from abc import ABC

import torch
import torch.nn.functional as F
from spert.beam import BeamSearch


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class SpERTLoss(Loss):
    def __init__(self, rel_criterion, entity_criterion, model, optimizer = None, scheduler = None, max_grad_norm = None):
        self._rel_criterion = rel_criterion
        self._entity_criterion = entity_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm
        self._device = model._device


    def compute(self, entity_logits, entity_labels, rel_logits, rel_labels, token_masks, is_eval=False):
        # entity loss

        entity_loss = torch.tensor(0., dtype=torch.float).to(self._device)
        rel_loss = torch.tensor(0., dtype=torch.float).to(self._device)

        if not is_eval:
            entity_labels = torch.masked_select(entity_labels, token_masks.sum(dim=1).bool())
            entity_logits = entity_logits[token_masks.sum(dim=2).bool()[:, :-1]]
            # print("entity labels:", entity_labels)
            # print("preds:", entity_logits.argmax(dim=1))
            loss =  self._entity_criterion(entity_logits, entity_labels)
            entity_loss += loss.sum()

            if rel_logits.nelement() != 0 and rel_labels.nelement() != 0:
                encoding_len = rel_logits.shape[-1]
                rel_token_masks = token_masks.sum(dim=1).unsqueeze(1).repeat(1, token_masks.shape[1],1)
                loss = self._rel_criterion(rel_logits, rel_labels[:, :encoding_len, :encoding_len])
                loss = loss * rel_token_masks[:, :encoding_len, :encoding_len]
                rel_mask = torch.triu(torch.ones_like(loss, dtype=torch.bool), diagonal=1)
                # print("labels:", rel_labels[:, :-1, :-1] * rel_token_masks[:, :-1, :-1] * rel_mask)
                # print("pred:", rel_logits.argmax(dim=1)  * rel_token_masks[:, :-1, :-1] * rel_mask)
                masked_loss = torch.masked_select(loss, rel_mask)
                rel_loss += masked_loss.sum()


        else:
            for b, batch_logits in enumerate(entity_logits):
                batch_entities = entity_labels[b]
                loss = self._entity_criterion(batch_logits.squeeze(0), batch_entities)
    #             print("ent loss:", loss)
                entity_loss += loss.sum()

            if rel_logits != [] and rel_labels != []:

                for b, batch_logits in enumerate(rel_logits):
                    rel_mask = torch.triu(torch.ones_like(rel_labels[b], dtype=torch.bool), diagonal=1)
                    
                    batch_labels = rel_labels[b] * rel_mask
                    if batch_labels.nelement() == 0:
                        continue
                    batch_loss = self._rel_criterion(batch_logits, batch_labels.unsqueeze(0))
                    rel_loss += batch_loss.sum() 


#         print("entity loss:", entity_loss, "rel loss:", rel_loss)    
        train_loss = entity_loss + rel_loss
        
        if not is_eval:
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
            self._optimizer.step()
            self._scheduler.step()
#             self._model.zero_grad()
        return torch.tensor([train_loss.item(), entity_loss.item(), rel_loss.item()])


class NERLoss(Loss):

    def __init__(self, rel_criterion, entity_criterion, model, optimizer=None, scheduler=None, max_grad_norm=None):
        self._rel_criterion = rel_criterion
        self._entity_criterion = entity_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm

    def compute(self, entity_logits, entity_labels, token_masks, is_eval=False):

        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        token_masks = token_masks.view(-1)
        entity_labels = entity_labels.view(-1)
        entity_loss = self._entity_criterion(entity_logits, entity_labels)
        entity_loss = entity_loss * token_masks

        train_loss = entity_loss.sum()

        if not is_eval:
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
            self._optimizer.step()
            self._scheduler.step()
            # self._model.zero_grad()
        return torch.tensor([train_loss.item()])
