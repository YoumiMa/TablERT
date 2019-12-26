import random
from abc import ABC, abstractmethod
from typing import List, Iterable

import time
import torch
from torch import multiprocessing

import numpy as np

from spert import util
from spert.entities import Dataset, Token

multiprocessing.set_sharing_strategy('file_system')


class TrainTensorBatch:
    def __init__(self, encodings: torch.tensor, ctx_masks: torch.tensor,
                 entity_types: torch.tensor, entity_labels: torch.tensor,
                 rel_types: torch.tensor, rel_labels:torch.tensor, tokens: List[Token]):
        self.encodings = encodings
        self.ctx_masks = ctx_masks

        self.entity_types = entity_types
        self.entity_labels = entity_labels

        self.rel_types = rel_types
        self.rel_labels = rel_labels

        self.tokens = tokens

    def to(self, device):
        encodings = self.encodings.to(device)
        ctx_masks = self.ctx_masks.to(device)

        entity_types = self.entity_types.to(device)
        entity_labels = self.entity_labels.to(device)

        rel_types = self.rel_types.to(device)
        rel_labels = self.rel_labels.to(device)

        return TrainTensorBatch(encodings, ctx_masks, entity_types, entity_labels,
                                rel_types, rel_labels, self.tokens)


class EvalTensorBatch:
    def __init__(self, encodings: torch.tensor, ctx_masks: torch.tensor,
                 entity_masks: torch.tensor, entity_sizes: torch.tensor, entity_spans: torch.tensor,
                 entity_sample_masks: torch.tensor, tokens: List[Token]):
        self.encodings = encodings
        self.ctx_masks = ctx_masks

        self.entity_masks = entity_masks
        self.entity_sizes = entity_sizes
        self.entity_spans = entity_spans
        self.entity_sample_masks = entity_sample_masks

        self.tokens = tokens

    def to(self, device):
        encodings = self.encodings.to(device)
        ctx_masks = self.ctx_masks.to(device)

        entity_masks = self.entity_masks.to(device)
        entity_sizes = self.entity_sizes.to(device)
        entity_spans = self.entity_spans.to(device)
        entity_sample_masks = self.entity_sample_masks.to(device)

        return EvalTensorBatch(encodings, ctx_masks, entity_masks, entity_sizes, entity_spans, entity_sample_masks, self.tokens)


class TrainTensorSample:
    def __init__(self, encoding: torch.tensor, ctx_mask: torch.tensor,
                 entity_types: torch.tensor, entity_labels: torch.tensor, 
                 rel_types: torch.tensor, rel_labels: torch.tensor, tokens: List[Token]):
        self.encoding = encoding
        self.ctx_mask = ctx_mask
        
        self.entity_types = entity_types
        self.entity_labels = entity_labels

        self.rel_types = rel_types
        self.rel_labels = rel_labels

        self.tokens = tokens


class EvalTensorSample:
    def __init__(self, encoding: torch.tensor, ctx_mask: torch.tensor, entity_masks: torch.tensor,
                 entity_sizes: torch.tensor, entity_spans: torch.tensor, tokens: List[Token]):
        self.encoding = encoding
        self.ctx_mask = ctx_mask

        self.entity_masks = entity_masks
        self.entity_sizes = entity_sizes
        self.entity_spans = entity_spans

        self.tokens = tokens

class Sampler:
    def __init__(self, processes: int, limit: int):
        # multiprocessing
        self._processes = processes
        self._limit = limit
        self._ctx = multiprocessing.get_context("spawn") if processes > 0 else None
        self._manager = self._ctx.Manager() if processes > 0 else None
        self._pool = self._ctx.Pool(processes=processes) if processes > 0 else None

    def create_train_sampler(self, dataset: Dataset, batch_size: int, context_size: int,
                             order: Iterable = None, truncate: bool = False):
        train_sampler = TrainSampler(dataset, batch_size, context_size, order, truncate,
                                     self._manager, self._pool, self._processes, self._limit)
        return train_sampler

    def create_eval_sampler(self, dataset: Dataset, batch_size: int, context_size: int,
                            order: Iterable = None, truncate: bool = False):
        eval_sampler = EvalSampler(dataset, batch_size, context_size,
                                   order, truncate, self._manager, self._pool, self._processes, self._limit)
        return eval_sampler

    def join(self):
        if self._processes > 0:
            self._pool.close()
            self._pool.join()


class BaseSampler(ABC):
    def __init__(self, mp_func, manager, pool, processes, limit):
        # multiprocessing
        self._mp_func = mp_func
        self._manager = manager
        self._pool = pool
        self._processes = processes

        # avoid large memory consumption (e.g. in case of slow evaluation)
        self._semaphore = self._manager.Semaphore(limit) if processes > 0 else None

        self._current_batch = 0
        self._results = None

    @property
    @abstractmethod
    def _batches(self) -> List:
        pass

    def __next__(self):
        if self._current_batch < len(self._batches):
            if self._processes > 0:
                # multiprocessing
                batch, _ = self._results.next()
                self._semaphore.release()
            else:
                # no multiprocessing
                batch, _ = self._mp_func(self._batches[self._current_batch])

            self._current_batch += 1
            return batch
        else:
            raise StopIteration

    def __iter__(self):
        if self._processes > 0:
            # multiprocessing
            self._results = self._pool.imap(self._mp_func, self._batches)
        return self


class TrainSampler(BaseSampler):
    def __init__(self, dataset, batch_size, context_size, order, truncate, manager, pool, processes, limit):
        super().__init__(_produce_train_batch, manager, pool, processes, limit)

        self._dataset = dataset
        self._batch_size = batch_size
        self._context_size = context_size

        batches = self._dataset.iterate_documents(self._batch_size, order=order, truncate=truncate)
        self._prep_batches = self._prepare(batches)

    def _prepare(self, batches):
        prep_batches = []

        for i, batch in enumerate(batches):
            prep_batches.append((i, batch, self._context_size, self._semaphore))

        return prep_batches

    @property
    def _batches(self):
        return self._prep_batches


class EvalSampler(BaseSampler):
    def __init__(self, dataset, batch_size, context_size,
                 order, truncate, manager, pool, processes, limit):
        super().__init__(_produce_eval_batch, manager, pool, processes, limit)

        self._dataset = dataset
        self._batch_size = batch_size
        self._context_size = context_size

        batches = self._dataset.iterate_documents(self._batch_size, order=order, truncate=truncate)
        self._prep_batches = self._prepare(batches)

    def _prepare(self, batches):
        prep_batches = []

        for i, batch in enumerate(batches):
            prep_batches.append((i, batch, self._context_size, self._semaphore))

        return prep_batches

    @property
    def _batches(self):
        return self._prep_batches


def _produce_train_batch(args):
    i, docs, context_size, semaphore = args

    if semaphore is not None:
        semaphore.acquire()

    samples = []
    for d in docs:
        sample = _create_train_sample(d, context_size)
        samples.append(sample)

    batch = _create_train_batch(samples)

    return batch, i


def _produce_eval_batch(args):
    i, docs, context_size, semaphore = args

    if semaphore is not None:
        semaphore.acquire()

    samples = []
    for d in docs:
        sample = _create_eval_sample(d, context_size)
        samples.append(sample)

    batch = _create_eval_batch(samples)
    return batch, i


def _create_train_sample(doc, context_size, shuffle = False):
    encoding = doc.encoding
    token_count = len(doc.tokens)

    # positive entities
    entity_spans, entity_types, entity_labels = [], [], []
    for e in doc.entities:
        entity_spans.append(e.span)
        entity_types.append(e.entity_type)
        entity_labels.append(create_entity_mask(*e.span, context_size).to(torch.long))       
        for i, t in enumerate(e.tokens):
            entity_labels[-1][t.span_start:t.span_end] = e.entity_labels[i].index

    entity_types = torch.tensor([e.index for e in entity_types], dtype=torch.long)
    entity_labels = torch.stack(entity_labels).sum(axis=0)

    # positive relations
    rel_spans, rel_types= [], []
    rel_labels = [[0 for j in range(i) ]  for i in range(context_size-1, 0, -1)]
    for rel in doc.relations:
        s1, s2 = rel.head_entity.span, rel.tail_entity.span
        rel_spans.append((s1, s2))
        rel_types.append(rel.relation_type)
        # rel_labels[rel.tail_entity.span]
        former = rel.head_entity if s1[0] < s2[0] else rel.tail_entity
        latter = rel.tail_entity if s1[0] < s2[0] else rel.head_entity
        for i in range(former.span[0], former.span[1]):
            for j in range(latter.span[0], latter.span[1]):
                rel_labels[i][j-i-1] = rel.relation_label.index


    rel_types = torch.tensor([r.index for r in rel_types], dtype=torch.long)
    rel_labels = torch.tensor(sum(rel_labels, []))

    # create tensors
    # token indices
    _encoding = encoding
    encoding = torch.zeros(context_size, dtype=torch.long)
    encoding[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # masking of tokens
    ctx_mask = torch.zeros(context_size, dtype=torch.bool)
    ctx_mask[:len(_encoding)] = 1

    return TrainTensorSample(encoding=encoding, ctx_mask=ctx_mask, 
                            entity_types=entity_types, entity_labels=entity_labels,
                            rel_types=rel_types, rel_labels=rel_labels, tokens=doc.tokens)


def _create_eval_sample(doc, max_span_size, context_size):
    encoding = doc.encoding
    token_count = len(doc.tokens)

    # create entity candidates
    entity_spans = []
    entity_masks = []
    entity_sizes = []

    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span
            entity_spans.append(span)
            entity_masks.append(create_entity_mask(*span, context_size))
            entity_sizes.append(size)

    # create tensors
    # token indices
    _encoding = encoding
    encoding = torch.zeros(context_size, dtype=torch.long)
    encoding[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # masking of tokens
    ctx_mask = torch.zeros(context_size, dtype=torch.bool)
    ctx_mask[:len(_encoding)] = 1

    # entities
    entity_masks = torch.stack(entity_masks)
    entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
    entity_spans = torch.tensor(entity_spans, dtype=torch.long)

    return EvalTensorSample(encoding=encoding, ctx_mask=ctx_mask, entity_masks=entity_masks,
                            entity_sizes=entity_sizes, entity_spans=entity_spans, tokens=doc.tokens)


def _create_train_batch(samples):
    batch_encodings = []
    batch_ctx_masks = []

    batch_entity_types = []
    batch_rel_types = []

    batch_entity_labels = []
    batch_rel_labels = []

    for sample in samples:
        encoding = sample.encoding
        ctx_mask = sample.ctx_mask

        # entities
        entity_types = sample.entity_types
        entity_labels = sample.entity_labels

        # relations
        rel_types = sample.rel_types
        rel_labels = sample.rel_labels


        batch_encodings.append(encoding)
        batch_ctx_masks.append(ctx_mask)

        batch_rel_types.append(rel_types)
        batch_entity_types.append(entity_types)
        
        batch_rel_labels.append(rel_labels)
        batch_entity_labels.append(entity_labels)

    # stack samples

    encodings = util.padded_stack(batch_encodings)
    ctx_masks = util.padded_stack(batch_ctx_masks)


    batch_rel_types = util.padded_stack(batch_rel_types)
    batch_entity_types = util.padded_stack(batch_entity_types)

    batch_rel_labels = util.padded_stack(batch_rel_labels)
    batch_entity_labels = util.padded_stack(batch_entity_labels)
    
    batch = TrainTensorBatch(encodings=encodings, ctx_masks=ctx_masks, 
                            entity_types=batch_entity_types, entity_labels=batch_entity_labels,
                            rel_types=batch_rel_types, rel_labels=batch_rel_labels, tokens=sample.tokens)

    return batch


def _create_eval_batch(samples):
    batch_encodings = []
    batch_ctx_masks = []

    batch_entity_masks = []
    batch_entity_sizes = []
    batch_entity_spans = []
    batch_entity_sample_masks = []

    for sample in samples:
        encoding = sample.encoding
        ctx_mask = sample.ctx_mask

        entity_masks = sample.entity_masks
        entity_sizes = sample.entity_sizes
        entity_spans = sample.entity_spans

        # tensors to mask entity samples of batch
        # since samples are stacked into batches, "padding" entities possibly must be created
        # these are later masked during evaluation
        entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)

        # corner case handling (no entities)
        if entity_masks.shape[0] == 0:
            entity_masks = torch.tensor([0 for i in range(entity_masks.shape[-1])], dtype=torch.bool)
            entity_sizes = torch.tensor([0], dtype=torch.long)
            entity_spans = torch.tensor([[0, 0]], dtype=torch.long)
            entity_sample_masks = torch.tensor([0], dtype=torch.bool)

        batch_encodings.append(encoding)
        batch_ctx_masks.append(ctx_mask)

        batch_entity_masks.append(entity_masks)
        batch_entity_sizes.append(entity_sizes)
        batch_entity_spans.append(entity_spans)
        batch_entity_sample_masks.append(entity_sample_masks)

    # stack samples
    encodings = util.padded_stack(batch_encodings)
    ctx_masks = util.padded_stack(batch_ctx_masks)

    batch_entity_masks = util.padded_stack(batch_entity_masks)
    batch_entity_sizes = util.padded_stack(batch_entity_sizes)
    batch_entity_spans = util.padded_stack(batch_entity_spans)
    batch_entity_sample_masks = util.padded_stack(batch_entity_sample_masks)

    batch = EvalTensorBatch(encodings=encodings, ctx_masks=ctx_masks, entity_masks=batch_entity_masks,
                            entity_sizes=batch_entity_sizes, entity_spans=batch_entity_spans,
                            entity_sample_masks=batch_entity_sample_masks, tokens=sample.tokens)

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
