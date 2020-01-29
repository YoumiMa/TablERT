from abc import ABC

import torch
from spert.beam import BeamSearch


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
        m = torch.nn.Softmax(dim=-1)
        score = m(entity_logits)
        # print("local:", score)

        return score

    def compute(self, entity_logits, entity_labels, rel_logits, rel_labels):
        # entity loss

        train_loss = 0.

        for b, batch_logits in enumerate(entity_logits):

            # batch_entities = entity_labels[b][1:1+batch_logits.shape[1]]
            batch_entities = entity_labels[b]
            # # print(batch_logits.shape)
            # entity_loss = self._entity_criterion(batch_logits.squeeze(0).squeeze(1), batch_entities)
            # train_loss += entity_loss.sum()/batch_logits.shape[1]

            context_size = batch_logits.shape[1]
            # print(context_size)
            local_scores = []
            beam_paths = []
            ptr = []
            beam = BeamSearch(batch_logits.shape[2])
            for i in range(context_size): # no [CLS], no [SEP]
                # print("logits:", batch_logits.squeeze(0)[i])
                logits = batch_logits.squeeze(0)[i]
                # print("softmaxed:", logits)
                beam.advance(logits)
                preds = beam.get_curr_state
                gold = batch_entities[i]
                # print("preds:", preds, "gold:", gold)
                local_scores.append(logits[0][gold])
                # print("beam:", beam.get_curr_scores)
                # print("sum:", torch.logsumexp(beam.get_curr_scores, dim=0))
                beam_paths.append(torch.logsumexp(beam.get_curr_scores, dim=0))
                # print(beam_paths)
                # print("curr scores:", beam_paths)
                if gold not in preds:
                    ptr.append(i)


            if ptr == []:
                ptr.append(context_size)
            # for p in ptr:
                # print("p:", p , - sum(local_scores[:p+1]) +  sum(greedy_path[:p+1]))

            ### Cumloss ###
            # for p in ptr:
            #     train_loss += - sum(local_scores[:p+1]) +  beam_paths[min(p, context_size-1)]

            ### final loss ###
            p = ptr[-1]
            train_loss += - sum(local_scores[:p+1]) +  beam_paths[min(p, context_size-1)]
            
            # print("loss:", train_loss)
            # train_loss = self._entity_criterion(batch_logits.squeeze(0), entity_labels[b][1:-1])
            # train_loss =  (train_loss * entity_mask[b][1:-1]).sum() / entity_mask[b][1:-1].sum()
            # train_loss /= context_size

            # print(local_scores, sum(local_scores))
            # train_loss += entity_loss

        if rel_logits != []:
            for b, batch_logits in enumerate(rel_logits):
                print(batch_logits.shape, rel_labels)
                batch_rels = rel_labels[b][1:1+batch_logits.shape[1]]
                context_size = batch_logits.shape[1]
                local_scores = []
                beam_paths = []
                ptr = []
                beam = BeamSearch(batch_logits.shape[2])
                for i in range(context_size): 
                    logits = batch_logits.squeeze(0)[i-1]
                    beam.advance(logits)
                    preds = beam.get_curr_state
                    gold = batch_entities[i]
                    # print("preds:", preds, "gold:", gold)
                    local_scores.append(logits[0][gold])
                    # print("beam:", beam.get_curr_scores)
                    # print("sum:", torch.logsumexp(beam.get_curr_scores, dim=0))
                    beam_paths.append(torch.logsumexp(beam.get_curr_scores, dim=0))
                    # print(beam_paths)
                    # print("curr scores:", beam_paths)
                    if gold not in preds:
                        ptr.append(i)


                if ptr == []:
                    ptr.append(context_size)
            # for p in ptr:
                # print("p:", p , - sum(local_scores[:p+1]) +  sum(greedy_path[:p+1]))
                
                # for p in ptr:
                p = ptr[-1]
                train_loss += - sum(local_scores[:p+1]) +  beam_paths[min(p, context_size-1)]

                # print(local_scores, sum(local_scores))             

        # print("loss:", train_loss)
        # print("=" * 50)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        # self._model.zero_grad()
        return train_loss.item()
