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

    def compute(self, entity_logits, curr_i, token_masks, entity_labels):
        # entity loss
        curr_token = token_masks[:, curr_i]
        batch_size = entity_labels.shape[0]

        curr_label = torch.masked_select(entity_labels, curr_token)
        curr_label = curr_label.view(batch_size, -1)
        curr_label = curr_label.unique(dim=-1)

        # print("logits:", entity_logits.argmax(dim=1))
        # print("curr_label:", curr_label)

        entity_loss = self._entity_criterion(entity_logits, curr_label.squeeze())
        entity_loss = entity_loss.sum() / entity_loss.shape[-1]
        # print("loss:", entity_loss)

        train_loss = entity_loss

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()
