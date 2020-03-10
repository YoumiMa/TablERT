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


    def compute(self, entity_logits, entity_labels, rel_logits, rel_labels, is_eval=False):
        # entity loss

        entity_loss = torch.tensor(0., dtype=torch.float).to(self._device)
        rel_loss = torch.tensor(0., dtype=torch.float).to(self._device)
        for b, batch_logits in enumerate(entity_logits):
            # batch_entities = entity_labels[b][1:1+batch_logits.shape[1]]
            batch_entities = entity_labels[b]
            # context_size = batch_entities.shape[-1]
            # print("labels:", batch_entities)
            # print("pred:", batch_logits.argmax(dim=2))
            # print(batch_logits.squeeze(0).shape)
            # print(batch_entities.shape)
            loss = self._entity_criterion(batch_logits.squeeze(0), batch_entities)
            entity_loss += loss.sum()


        if rel_logits != [] and rel_labels != []:

            for b, batch_logits in enumerate(rel_logits):
                rel_mask = torch.triu(torch.ones_like(rel_labels[b], dtype=torch.bool), diagonal=1)
                
                batch_labels = torch.masked_select(rel_labels[b], rel_mask)
                if batch_labels.nelement() == 0:
                    continue
                batch_logits = batch_logits[:, rel_mask]
                batch_logits = batch_logits.view(-1, batch_logits.shape[-1])

                # print("rel labels:", batch_labels)
                # print("pred:", batch_logits.argmax(dim=1))
                # print("labels:", batch_labels)
                # local_scores = batch_logits[torch.arange(batch_labels.shape[0]), batch_labels]
                # print("local scores:", local_scores)
                # beam_scores, preds = batch_logits.topk(k=beam.get_beam_size, dim=1, largest=True, sorted=True)
                # print("beam scores", beam_scores)
                # gold_in_beam = (preds == batch_labels.unsqueeze(1).repeat(1,beam.get_beam_size)).sum(dim=1)
                # wrong_preds = (gold_in_beam == 0).nonzero()
                # print("wrong preds:", wrong_preds)

                # p = batch_labels.shape[-1] if wrong_preds.nelement() == 0 else wrong_preds[-1].item()

                # rel_loss += - sum(local_scores[:p+1]) + torch.logsumexp(beam_scores[:p+1].sum(dim=0), dim=0)
                # print("rel loss:", rel_loss, "pos:",sum(local_scores[:p+1]), "neg:",  torch.logsumexp(beam_scores[:p+1].sum(dim=0), dim=0))

                batch_loss = self._rel_criterion(batch_logits, batch_labels)
                rel_loss += batch_loss.sum() 


        # print("entity loss:", entity_loss, "rel loss:", rel_loss)    
        train_loss = entity_loss + rel_loss
        
        if not is_eval:
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
            self._optimizer.step()
            self._scheduler.step()
            # self._model.zero_grad()
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
