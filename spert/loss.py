from abc import ABC

import torch


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class SpERTLoss(Loss):
    def __init__(self, rel_criterion, entity_criterion, model, optimizer, scheduler, max_grad_norm):
        self._rel_criterion = rel_criterion
        self._entity_criterion = entity_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm

    def compute(self, entity_logits, entity_labels, curr_i):
        # entity loss

        curr_entity_labels = entity_labels.clone()
        curr_entity_labels = curr_entity_labels[:, :curr_i+1].view(curr_i+1)
        curr_entity_logits = entity_logits.clone()
        curr_entity_logits = curr_entity_logits[:, :curr_i+1, :].view(curr_i+1, curr_entity_logits.shape[2])
        # print("logits:", curr_entitsy_logits.argmax(dim=1))
        # print("gold:", curr_entity_labels)

        prev_loss =  self._entity_criterion(curr_entity_logits, curr_entity_labels)
        prev_loss = prev_loss.sum() / prev_loss.shape[-1]
        # print(curr_entity_logits[-1, :].unsqueeze(0), curr_entity_labels[-1])
        entity_loss = self._entity_criterion(curr_entity_logits[-1, :].unsqueeze(0), curr_entity_labels[-1].unsqueeze(0))
        entity_loss = entity_loss.sum() / entity_loss.shape[-1]
        # print("loss:", entity_loss)

        train_loss = entity_loss

        train_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()
