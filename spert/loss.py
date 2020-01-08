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

    def compute(self, entity_logits, entity_labels, rel_logits, rel_labels, entity_mask):
        # entity loss

        train_loss = 0.

        for b, batch_logits in enumerate(entity_logits):
            context_size = entity_mask[b].sum()
            # print(context_size)
            local_scores = []
            greedy_path = []
            ptr = []
            for i in range(1, context_size-1): # no [CLS], no [SEP]
                logits = batch_logits.squeeze(0)[i-1]
                pred = torch.argmax(logits)
                gold = entity_labels[b][i]

                # score = self.local_score(logits)
                # print("score:", score)
                local_scores.append(logits[gold])
                greedy_path.append(logits[pred])

                if pred == gold:
                    ptr.append(i-1)
            if ptr == []:
                ptr.append(context_size-2-1)
            # entity_loss = - sum(local_scores)/sum(greedy_path) * ptr
            for p in ptr:
                train_loss += - sum(local_scores[:p+1]) /sum(greedy_path[:p+1])
            # train_loss = self._entity_criterion(batch_logits.squeeze(0), entity_labels[b][1:-1])
            # train_loss =  (train_loss * entity_mask[b][1:-1]).sum() / entity_mask[b][1:-1].sum()
            # train_loss /= context_size

            # # print(local_scores, sum(local_scores))
            # train_loss += entity_loss

        if rel_logits != []:
            for b, batch_logits in enumerate(rel_logits):
                # print(batch_logits.shape, rel_labels)
                context_size = entity_mask[b].sum()
                local_scores = []
                greedy_path = []
                ptr = []
                for i in range(1, context_size-2): # no [CLS], no [SEP]
                    logits = batch_logits.squeeze(0)[i-1]
                    pred = torch.argmax(logits)
                    gold = rel_labels[b][i]
                    # print("pred:", pred, "gold:", gold)
                    # score = self.local_score(logits)
                    # print("score:", score)
                    local_scores.append(logits[gold])
                    greedy_path.append(logits[pred])

                    if pred == gold:
                        ptr.append(i-1)

                if ptr == []:
                    ptr.append(context_size-4)
                for p in ptr:
                    train_loss += - sum(local_scores[:p+1]) + sum(greedy_path[:p+1])

                # print(local_scores, sum(local_scores))             

        # print("loss:", train_loss)
        # print("=" * 50)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        # self._model.zero_grad()
        return train_loss.item()
