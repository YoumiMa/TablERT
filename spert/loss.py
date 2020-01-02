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


    def local_score(self, entity_logits, label_mask):
        m = torch.nn.Softmax()
        score = m(entity_logits[label_mask])
        # print("local:", score)

        return score

    def compute(self, score, entity_logits, curr_token, label_mask, entity_labels):
        # entity loss


        batch_size = entity_labels.shape[0]
        prev_labels = []
        curr_labels = []

        local_score = self.local_score(entity_logits, label_mask)
        score += local_score
        # print("global:", score)
        for i in range(batch_size):
            curr_label = entity_labels[i, curr_token[i]]
            curr_label = curr_label.unique()
            curr_labels.append(curr_label)   

        curr_labels = torch.cat(curr_labels)


        print("=" * 50)

        beam_sum = torch.sort(score, dim=1, descending=True)[0]
        # print(beam_sum[:,:2].sum(dim=1))
        # print(score/beam_sum[:,:2].sum(dim=1))
        entity_loss = self._entity_criterion(local_score/score, curr_labels)
        entity_loss = entity_loss.sum() / entity_loss.shape[-1]
        print("logits:", torch.sort(score, dim=1, descending=True))
        print("curr_label:", curr_labels)
        train_loss = entity_loss
        print("loss:", train_loss)
        train_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        # self._model.zero_grad()
        return train_loss.item(), score
