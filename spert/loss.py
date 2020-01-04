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


    def local_score(self, entity_logits):
        m = torch.nn.LogSoftmax()
        score = m(entity_logits)
        # print("local:", score)

        return score

    def compute(self, entity_logits, entity_labels, rel_logits, rel_labels):
        # entity loss

        train_loss = 0.

        for b, batch_logits in enumerate(entity_logits):
            context_size = entity_labels.shape[-1]
            local_scores = []
            greedy_path = []
            ptr = 0
            for i in range(1, context_size-1): # no [CLS], no [SEP]
                logits = batch_logits.squeeze(0)[i-1]
                pred = torch.argmax(logits)
                gold = entity_labels[b][i]
                # print("pred:", pred, "gold:", gold)
                # score = self.local_score(logits)
                # print("score:", score)
                local_scores.append(logits[gold])
                greedy_path.append(logits[pred])

                if pred != gold:
                    ptr = i

            # entity_loss = - sum(local_scores)/sum(greedy_path) * ptr
            entity_loss = - sum(local_scores)/sum(greedy_path)

            # print(local_scores, sum(local_scores))
            train_loss += entity_loss

        if rel_logits != []:
            for b, batch_logits in enumerate(rel_logits):
                # print(batch_logits.shape, rel_labels)
                context_size = rel_labels.shape[-1]
                local_scores = []
                greedy_path = []
                ptr = 0
                for i in range(1, context_size-2): # no [CLS], no [SEP]
                    logits = batch_logits.squeeze(0)[i]
                    pred = torch.argmax(logits)
                    gold = rel_labels[b][i]
                    # print("pred:", pred, "gold:", gold)
                    # score = self.local_score(logits)
                    # print("score:", score)
                    local_scores.append(logits[gold])
                    greedy_path.append(logits[pred])

                    if pred != gold:
                        ptr = i

                # entity_loss = - sum(local_scores)/sum(greedy_path) * ptr
                rel_loss = - sum(local_scores)/sum(greedy_path)

                # print(local_scores, sum(local_scores))
                train_loss += rel_loss                

        print("loss:", train_loss)
        # print("=" * 50)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        # self._model.zero_grad()
        return train_loss.item()
