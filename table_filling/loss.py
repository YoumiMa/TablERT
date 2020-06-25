from abc import ABC

import torch
import torch.nn.functional as F


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class TableLoss(Loss):
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

        for b, batch_logits in enumerate(entity_logits):
            batch_entities = entity_labels[b]
            loss = self._entity_criterion(batch_logits.squeeze(0), batch_entities)

            entity_loss += loss.sum()

        if rel_logits != [] and rel_labels != []:
            for b, batch_logits in enumerate(rel_logits):
                batch_labels = rel_labels[b]
                if batch_labels.nelement() == 0:
                    continue
                batch_loss = self._rel_criterion(batch_logits, batch_labels.unsqueeze(0))

                batch_loss_masked = torch.triu(batch_loss, diagonal=1)

                rel_loss += batch_loss_masked.sum() 

 
        train_loss =  entity_loss + rel_loss
        
        if not is_eval:
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
            self._optimizer.step()
            self._scheduler.step()
            self._model.zero_grad()
        return torch.tensor([train_loss.item(), entity_loss.item(), rel_loss.item()])


