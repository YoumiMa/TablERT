import random
from abc import ABC, abstractmethod
from typing import List, Iterable

import time
import torch
from torch import multiprocessing

import numpy as np

from table_filling import util
from table_filling.entities import Dataset, Token

multiprocessing.set_sharing_strategy('file_system')


class TensorBatch:
    def __init__(self, encodings: torch.tensor, ctx_masks: torch.tensor, 
                 entity_masks: List[torch.tensor],
                 entity_labels: torch.tensor, rel_labels:torch.tensor, 
                 token_masks: torch.tensor, start_token_masks: torch.tensor):
        
        self.encodings = encodings
        self.ctx_masks = ctx_masks

        self.entity_masks = entity_masks
        self.entity_labels = entity_labels

        self.rel_labels = rel_labels

        self.token_masks = token_masks
        self.start_token_masks = start_token_masks
        
    def to(self, device):
        encodings = self.encodings.to(device)
        ctx_masks = self.ctx_masks.to(device)

        entity_masks = [m.to(device) for m in self.entity_masks]
        entity_labels = self.entity_labels.to(device)

        rel_labels = self.rel_labels.to(device)

        token_masks = self.token_masks.to(device)
        start_token_masks = self.start_token_masks.to(device)

        return TensorBatch(encodings, ctx_masks, entity_masks, entity_labels, rel_labels, token_masks, start_token_masks)
    
    

class TrainTensorBatch(TensorBatch):
    def __init__(self, encodings: torch.tensor, ctx_masks: torch.tensor, 
                 entity_masks: List[torch.tensor],
                 entity_labels: torch.tensor, rel_labels:torch.tensor, 
                 token_masks: torch.tensor, start_token_masks: torch.tensor):
        
        super().__init__(encodings, ctx_masks, entity_masks, 
                         entity_labels, rel_labels, 
                         token_masks, start_token_masks)


        
class EvalTensorBatch(TensorBatch):
    def __init__(self, encodings: torch.tensor, ctx_masks: torch.tensor,
                 entity_masks: List[torch.tensor],
                 entity_labels: torch.tensor, rel_labels:torch.tensor, 
                 token_masks: torch.tensor, start_token_masks: torch.tensor):
        
        super().__init__(encodings, ctx_masks, entity_masks, 
                         entity_labels, rel_labels, 
                         token_masks, start_token_masks)


class TensorSample:
    def __init__(self, encoding: torch.tensor, ctx_mask: torch.tensor,
                 entity_masks: torch.tensor,
                 entity_labels: torch.tensor, rel_labels: torch.tensor, 
                 token_masks: torch.tensor, start_token_masks: torch.tensor):
        self.encoding = encoding
        self.ctx_mask = ctx_mask

        self.entity_masks = entity_masks
        self.entity_labels = entity_labels

        self.rel_labels = rel_labels

        self.token_masks = token_masks
        self.start_token_masks = start_token_masks
        

class TrainTensorSample(TensorSample):
    def __init__(self, encoding: torch.tensor, ctx_mask: torch.tensor,
                 entity_masks: torch.tensor,
                 entity_labels: torch.tensor, rel_labels: torch.tensor, 
                 token_masks: torch.tensor, start_token_masks: torch.tensor):
        
        super().__init__(encoding, ctx_mask, entity_masks,
                        entity_labels, rel_labels, token_masks, start_token_masks)


class EvalTensorSample(TensorSample):
    def __init__(self, encoding: torch.tensor, ctx_mask: torch.tensor,
                 entity_masks: torch.tensor,
                 entity_labels: torch.tensor, rel_labels: torch.tensor, 
                 token_masks: torch.tensor, start_token_masks: torch.tensor):
        
        super().__init__(encoding, ctx_mask, entity_masks,
                        entity_labels, rel_labels, token_masks, start_token_masks)

        
class Sampler:
    def __init__(self):
        return

    def create_train_sampler(self, dataset: Dataset, batch_size: int, context_size: int,
                             order: Iterable = None, truncate: bool = False):
        train_sampler = TrainSampler(dataset, batch_size, context_size, order, truncate)
        return train_sampler

    def create_eval_sampler(self, dataset: Dataset, batch_size: int, context_size: int,
                            order: Iterable = None, truncate: bool = False):
        eval_sampler = EvalSampler(dataset, batch_size, context_size, order, truncate)
        return eval_sampler



class BaseSampler(ABC):
    def __init__(self, mp_func):
        # multiprocessing
        self._mp_func = mp_func

        self._current_batch = 0
        self._results = None

    @property
    @abstractmethod
    def _batches(self) -> List:
        pass

    def __next__(self):
        if self._current_batch < len(self._batches):
            batch, _ = self._mp_func(self._batches[self._current_batch])

            self._current_batch += 1
            return batch
        else:
            raise StopIteration

    def __iter__(self):
        return self
        

class TrainSampler(BaseSampler):
    def __init__(self, dataset, batch_size, context_size, order, truncate):
        
        super().__init__(_produce_train_batch)
        
        self._dataset = dataset
        self._batch_size = batch_size
        self._context_size = context_size

        batches = self._dataset.iterate_documents(self._batch_size, order=order, truncate=truncate)
        self._prep_batches = self._prepare(batches)

    def _prepare(self, batches):
        prep_batches = []

        for i, batch in enumerate(batches):
            prep_batches.append((i, batch, self._context_size))

        return prep_batches

    @property
    def _batches(self):
        return self._prep_batches


class EvalSampler(BaseSampler):
    def __init__(self, dataset, batch_size, context_size, order, truncate):
        
        super().__init__(_produce_eval_batch)
        
        self._dataset = dataset
        self._batch_size = batch_size
        self._context_size = context_size

        batches = self._dataset.iterate_documents(self._batch_size, order=order, truncate=truncate)
        self._prep_batches = self._prepare(batches)

    def _prepare(self, batches):
        prep_batches = []

        for i, batch in enumerate(batches):
            prep_batches.append((i, batch, self._context_size))

        return prep_batches

    @property
    def _batches(self):
        return self._prep_batches


def _produce_train_batch(args):

    i, docs, context_size = args
    
    samples = []
    for d in docs:
        sample = _create_train_sample(d, context_size)
        samples.append(sample)

    batch = _create_train_batch(samples)

    return batch, i


def _produce_eval_batch(args):
    i, docs, context_size = args

    samples = []
    for d in docs:
        sample = _create_eval_sample(d, context_size)
        samples.append(sample)

    batch = _create_eval_batch(samples)
    return batch, i


def _create_train_sample(doc, context_size, shuffle = False):
    encoding = doc.encoding
#     print(doc.doc_id)
    token_count = len(doc.tokens)

    # positive entities
    entity_labels = []

    entity_masks = torch.zeros((token_count + 2, token_count + 2), dtype=torch.bool)

    entity_masks.fill_diagonal_(1)

    for e in doc.entities:
        entity_labels.append(create_entity_mask(*e.span, context_size).to(torch.long))  
        entity_masks[e.tokens[0].index + 1 :e.tokens[-1].index + 2, e.tokens[0].index + 1 :e.tokens[-1].index + 2] = 1     
        for i, t in enumerate(e.tokens):
            entity_labels[-1][t.span_start:t.span_end] = e.entity_labels[i].index


    if not doc.entities: # no entities included
        entity_labels = torch.zeros(context_size, dtype=torch.long)
    else:
        entity_labels = torch.stack(entity_labels).sum(axis=0)

    # positive relations
    rel_spans = []
    rel_labels = torch.zeros((context_size, context_size), dtype=torch.long)

    for rel in doc.relations:
        s1, s2 = rel.head_entity.span, rel.tail_entity.span
        rel_spans.append((s1, s2))
        # rel_labels[rel.tail_entity.span]
        former = rel.head_entity if s1[0] < s2[0] else rel.tail_entity
        latter = rel.tail_entity if s1[0] < s2[0] else rel.head_entity

        # ## map to all words in an entity.
        for i in range(former.span[0], former.span[1]):
            for j in range(latter.span[0], latter.span[1]):
                rel_labels[i][j] = rel.relation_label.index
        ### map to last word in an entity.
        # for i in range(former._tokens[-1].span_start, former._tokens[-1].span_end):
        #     for j in range(latter._tokens[-1].span_start, latter._tokens[-1].span_end):
        #         rel_labels[i][j] = rel.relation_label.index        

    # create tensors
    # token indices
    _encoding = encoding
    encoding = torch.zeros(context_size, dtype=torch.long)
    encoding[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # context mask
    ctx_mask = torch.zeros(context_size, dtype=torch.bool)
    ctx_mask[:len(_encoding)] = 1

    # token masks
    tokens = doc.tokens
    token_masks = torch.zeros((len(_encoding), context_size), dtype=torch.bool)
    start_token_masks = torch.zeros((len(_encoding), context_size), dtype=torch.bool)

    # [CLS]
    token_masks[0,0] = 1

    for i,t in enumerate(tokens):
        token_masks[i+1, t.span_start:t.span_end] = 1
#         print(t.span_start, t.span_end)
        start_token_masks[i+1, t.span_start] = 1
    # [SEP]
    token_masks[i+2,t.span_end] = 1
#     print(entity_labels)
    return TrainTensorSample(encoding=encoding, ctx_mask=ctx_mask, entity_masks=entity_masks,
                            entity_labels=entity_labels, rel_labels=rel_labels, 
                            token_masks=token_masks, start_token_masks=start_token_masks)


def _create_eval_sample(doc, context_size):
    encoding = doc.encoding
    token_count = len(doc.tokens)

    # positive entities
    entity_labels = []

    entity_masks = torch.zeros((token_count + 2, token_count + 2), dtype=torch.bool)

    entity_masks.fill_diagonal_(1)


    for e in doc.entities:
#         print(e.phrase, [t for t in e.tokens], e.entity_labels)
        entity_labels.append(create_entity_mask(*e.span, context_size).to(torch.long))       
        entity_masks[e.tokens[0].index + 1 :e.tokens[-1].index + 2, e.tokens[0].index + 1 : e.tokens[-1].index + 2] = 1     
        for i, t in enumerate(e.tokens):
            entity_labels[-1][t.span_start:t.span_end] = e.entity_labels[i].index

    if not doc.entities: # no entities included
        entity_labels = torch.zeros(context_size, dtype=torch.long)
    else:
        entity_labels = torch.stack(entity_labels).sum(axis=0)

    # positive relations
    rel_spans = []
    rel_labels = torch.zeros((context_size, context_size), dtype=torch.long)
    for rel in doc.relations:
        s1, s2 = rel.head_entity.span, rel.tail_entity.span
        rel_spans.append((s1, s2))

        former = rel.head_entity if s1[0] < s2[0] else rel.tail_entity
        latter = rel.tail_entity if s1[0] < s2[0] else rel.head_entity
        for i in range(former.span[0], former.span[1]):
            for j in range(latter.span[0], latter.span[1]):
                rel_labels[i][j] = rel.relation_label.index

        # for i in range(former._tokens[-1].span_start, former._tokens[-1].span_end):
        #     for j in range(latter._tokens[-1].span_start, latter._tokens[-1].span_end):
        #         rel_labels[i][j] = rel.relation_label.index 



    # create tensors
    # token indices
    _encoding = encoding
    encoding = torch.zeros(context_size, dtype=torch.long)
    encoding[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # context mask
    ctx_mask = torch.zeros(context_size, dtype=torch.bool)
    ctx_mask[:len(_encoding)] = 1

    # token masks
    tokens = doc.tokens
    token_masks = torch.zeros((len(_encoding), context_size), dtype=torch.bool)
    start_token_masks = torch.zeros((len(_encoding), context_size), dtype=torch.bool)

    # [CLS]
    token_masks[0,0] = 1

    
    for i,t in enumerate(tokens):
        token_masks[i+1, t.span_start:t.span_end] = 1
        start_token_masks[i+1, t.span_start] = 1

    # [SEP]
    token_masks[i+2,t.span_end] = 1           

    return EvalTensorSample(encoding=encoding, ctx_mask=ctx_mask, entity_masks=entity_masks,
                            entity_labels=entity_labels, rel_labels=rel_labels, 
                            token_masks=token_masks, start_token_masks=start_token_masks)



def _create_train_batch(samples):
    batch_encodings = []
    batch_ctx_masks = []

    batch_entity_masks = []

    batch_entity_labels = []
    batch_rel_labels = []

    batch_token_masks = []
    batch_start_token_masks = []


    for sample in samples:
        encoding = sample.encoding
        ctx_mask = sample.ctx_mask

        # entities
        entity_masks = sample.entity_masks
        entity_labels = sample.entity_labels

        # relations
        rel_labels = sample.rel_labels

        # token masks
        token_masks = sample.token_masks
        start_token_masks = sample.start_token_masks

        batch_encodings.append(encoding)
        batch_ctx_masks.append(ctx_mask)

        
        batch_rel_labels.append(rel_labels)
        batch_entity_labels.append(entity_labels)

        batch_entity_masks.append(entity_masks)

        batch_token_masks.append(token_masks)
        batch_start_token_masks.append(start_token_masks)

    # stack samples

    encodings = util.padded_stack(batch_encodings)
    ctx_masks = util.padded_stack(batch_ctx_masks)



    batch_rel_labels = util.padded_stack(batch_rel_labels)
    batch_entity_labels = util.padded_stack(batch_entity_labels)

    batch_token_masks = util.padded_stack(batch_token_masks)
    batch_start_token_masks = util.padded_stack(batch_start_token_masks)
    
    batch = TrainTensorBatch(encodings=encodings, ctx_masks=ctx_masks,
                            entity_masks=batch_entity_masks, 
                            entity_labels=batch_entity_labels, rel_labels=batch_rel_labels, 
                            token_masks=batch_token_masks, start_token_masks=batch_start_token_masks)

    return batch


def _create_eval_batch(samples):
    batch_encodings = []
    batch_ctx_masks = []

    batch_entity_masks = []

    batch_entity_labels = []
    batch_rel_labels = []

    batch_token_masks = []
    batch_start_token_masks = []

    for sample in samples:
        encoding = sample.encoding
        ctx_mask = sample.ctx_mask

        # entities
        entity_masks = sample.entity_masks
        entity_labels = sample.entity_labels

        # relations
        rel_labels = sample.rel_labels

        # token masks
        token_masks = sample.token_masks
        start_token_masks = sample.start_token_masks

        batch_encodings.append(encoding)
        batch_ctx_masks.append(ctx_mask)

        batch_rel_labels.append(rel_labels)
        batch_entity_labels.append(entity_labels)

        batch_entity_masks.append(entity_masks)


        batch_token_masks.append(token_masks)
        batch_start_token_masks.append(start_token_masks)

    # stack samples

    encodings = util.padded_stack(batch_encodings)
    ctx_masks = util.padded_stack(batch_ctx_masks)


    batch_rel_labels = util.padded_stack(batch_rel_labels)
    batch_entity_labels = util.padded_stack(batch_entity_labels)

    batch_token_masks = util.padded_stack(batch_token_masks)
    batch_start_token_masks = util.padded_stack(batch_start_token_masks)
    
    batch = EvalTensorBatch(encodings=encodings, ctx_masks=ctx_masks, 
                            entity_masks=batch_entity_masks,
                            entity_labels=batch_entity_labels, rel_labels=batch_rel_labels, 
                            token_masks=batch_token_masks, start_token_masks=batch_start_token_masks)

    return batch



def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def create_rel_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_entity_mask(start, end, context_size)
    return mask