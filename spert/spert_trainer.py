import argparse
import math
import os

import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
import transformers
from transformers import AdamW
from transformers import BertTokenizer

from spert import models
from spert.entities import Dataset
from spert.evaluator import Evaluator
from spert.input_reader import JsonInputReader, BaseInputReader
from spert.loss import SpERTLoss, NERLoss, Loss
from spert.beam import BeamSearch
from tqdm import tqdm
from spert.sampling import Sampler
from spert.trainer import BaseTrainer
from spert import util

from typing import List

import math

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def align_label(entity: torch.tensor, rel: torch.tensor, token_mask: torch.tensor):
    """ Align tokenized label to word-piece label, masked by token_mask. """

    batch_size = entity.shape[0]
    token_count = token_mask.to(torch.bool).sum()
    # print("entity:", entity)
    batch_entity_labels = []
    batch_rel_labels = []
    for b in range(batch_size):
        batch_entity_labels.append(torch.masked_select(entity[b], token_mask[b].sum(dim=0).to(torch.bool)))
        rel_ = rel[b][token_mask[b].sum(dim=0).to(torch.bool)]
        batch_rel_labels.append(rel_.t()[token_mask[b].sum(dim=0).to(torch.bool)].t())
    return batch_entity_labels, batch_rel_labels



class SpERTTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # byte-pair encoding
        self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                        do_lower_case=args.lowercase,
                                                        cache_dir=args.cache_path)

        # path to NER evalution output of BIO tagging scheme.

        self._bio_file_path = args.bio_file_path

        # path to export relation extraction examples to
        self._examples_path = os.path.join(self._log_path, 'examples_%s_%s_epoch_%s.html')

        # sampler (create and batch training/evaluation samples)
        self._sampler = Sampler(processes=args.sampling_processes, limit=args.sampling_limit)

        self._best_results['ner_micro_f1'] = 0
        self._best_results['rel_micro_f1'] = 0
        self._best_results['rel_ner_micro_f1'] = 0

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        train_label, valid_label = 'train', 'valid'

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
        self._logger.info("Model type: %s" % args.model_type)

        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._bio_file_path, self._tokenizer, self._logger)
        input_reader.read({train_label: train_path, valid_label: valid_path})
        self._log_datasets(input_reader)

        train_dataset = input_reader.get_dataset(train_label)
        train_sample_count = train_dataset.document_count
        updates_epoch = train_sample_count // args.train_batch_size
        updates_total = updates_epoch * args.epochs

        validation_dataset = input_reader.get_dataset(valid_label)

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)

        # create model
        model_class = models.get_model(self.args.model_type)

        # load model
        if args.model_type == 'table_filling':
            model = model_class.from_pretrained(self.args.model_path,
                                            cache_dir=self.args.cache_path,
                                            tokenizer= self._tokenizer,
                                            # SpERT model parameters
                                            relation_labels= input_reader.relation_label_count,
                                            entity_labels= input_reader.entity_label_count,
                                            max_entity_len = input_reader.max_entity_len,
                                            att_hidden = self.args.att_hidden,
                                            rnn_hidden = self.args.rnn_hidden,
                                            prop_drop=self.args.prop_drop,
                                            entity_label_embedding=self.args.entity_label_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            device=self._device)
        elif args.model_type == 'bert_ner':
            model = model_class.from_pretrained(self.args.model_path,
                                            cache_dir=self.args.cache_path,
                                            tokenizer= self._tokenizer,
                                            # SpERT model parameters
                                            relation_labels=input_reader.relation_label_count,
                                            entity_labels=input_reader.entity_label_count,
                                            prop_drop=self.args.prop_drop,
                                            entity_label_embedding=self.args.entity_label_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            device=self._device)

        # SpERT is currently optimized on a single GPU and not thoroughly tested in a multi GPU setup
        # If you still want to train SpERT on multiple GPUs, uncomment the following lines
        # # parallelize model
        # if self._device.type != 'cpu':
        #     model = torch.nn.DataParallel(model)

        model.to(self._device)

        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        # create scheduler

        if args.scheduler == 'constant':
            scheduler = transformers.get_constant_schedule(optimizer)
        elif args.scheduler == 'constant_warmup':
            scheduler = transformers.get_constant_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total)
        elif args.scheduler == 'linear_warmup':
            scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                     num_warmup_steps=args.lr_warmup * updates_total,
                                                                     num_training_steps=updates_total)
        elif args.scheduler == 'cosine_warmup':
            scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                                     num_warmup_steps=args.lr_warmup * updates_total,
                                                                     num_training_steps=updates_total)            
        elif args.scheduler == 'cosine_warmup_restart':
            scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                     num_warmup_steps=args.lr_warmup * updates_total,
                                                                     num_training_steps=updates_total,
                                                                     num_cycles= args.num_cycles)            


        # create loss function
        rel_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')

        if args.model_type == 'table_filling':
            compute_loss = SpERTLoss(rel_criterion, entity_criterion, model, optimizer, scheduler, args.max_grad_norm)
        elif args.model_type == 'bert_ner':
            compute_loss = NERLoss(rel_criterion, entity_criterion, model, optimizer, scheduler, args.max_grad_norm)

        # eval validation set
        if args.init_eval:
            self._eval(model, compute_loss, validation_dataset, input_reader, 0, updates_epoch)

        # train
        for epoch in range(args.epochs):
            # train epoch
            self._train_epoch(model, compute_loss, optimizer, train_dataset, updates_epoch, epoch,
                              input_reader.context_size, input_reader.entity_label_count, input_reader.relation_label_count,
                              input_reader._start_entity_label)

            # eval validation sets
            if not args.final_eval or (epoch == args.epochs - 1):
                ner_acc, rel_acc, rel_ner_acc = self._eval(model, compute_loss, validation_dataset, input_reader, epoch, updates_epoch)     
                if args.save_best:
                    extra = dict(epoch=epoch, updates_epoch=updates_epoch, epoch_iteration=0)
                    self._save_best(model=model, optimizer=optimizer if self.args.save_optimizer else None, 
                        accuracy=ner_acc[2], iteration=epoch * updates_epoch, label='ner_micro_f1', extra=extra)

        # save final model
        extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
        global_iteration = args.epochs * updates_epoch
        self._save_model(self._save_path, model, global_iteration,
                         optimizer=optimizer if self.args.save_optimizer else None, extra=extra,
                         include_iteration=False, name='final_model')

        self._logger.info("Logged in: %s" % self._log_path)
        self._logger.info("Saved in: %s" % self._save_path)

        self._sampler.join()

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)
        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._bio_file_path, self._tokenizer, self._logger)
        input_reader.read({dataset_label: dataset_path})
        self._log_datasets(input_reader)

        # create model
        model_class = models.get_model(self.args.model_type)

        # load model
        if args.model_type == 'table_filling':
            model = model_class.from_pretrained(self.args.model_path,
                                            cache_dir=self.args.cache_path,
                                            tokenizer= self._tokenizer,
                                            # SpERT model parameters
                                            relation_labels=input_reader.relation_label_count,
                                            entity_labels=input_reader.entity_label_count,
                                            max_entity_len = input_reader.max_entity_len,
                                            att_hidden = self.args.att_hidden,
                                            rnn_hidden = self.args.rnn_hidden,
                                            prop_drop=self.args.prop_drop,
                                            entity_label_embedding=self.args.entity_label_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            device=self._device)
        elif args.model_type == 'bert_ner':
            model = model_class.from_pretrained(self.args.model_path,
                                            cache_dir=self.args.cache_path,
                                            tokenizer= self._tokenizer,
                                            # SpERT model parameters
                                            relation_labels=input_reader.relation_label_count,
                                            entity_labels=input_reader.entity_label_count,
                                            prop_drop=self.args.prop_drop,
                                            entity_label_embedding=self.args.entity_label_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            device=self._device)

        model.to(self._device)

        # create loss function
        rel_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')

        if args.model_type == 'table_filling':
            compute_loss = SpERTLoss(rel_criterion, entity_criterion, model)
        elif args.model_type == 'bert_ner':
            compute_loss = NERLoss(rel_criterion, entity_criterion, model)

        # evaluate
        self._eval(model, compute_loss, input_reader.get_dataset(dataset_label), input_reader)
        self._logger.info("Logged in: %s" % self._log_path)

        self._sampler.join()

    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, 
                    optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int, context_size: int, 
                     entity_labels_count:int, relation_labels_count: int,
                     start_labels: List[int]):
        self._logger.info("Train epoch: %s" % epoch)

        # sort data according to context size
        # length_lst = [len(doc.encoding) for doc in dataset.documents]
        # order = sorted(range(len(length_lst)), key=lambda k: length_lst[k])

        # order = torch.randperm(dataset.document_count)
        order = None

        sampler = self._sampler.create_train_sampler(dataset, self.args.train_batch_size,
                                                     context_size, order=order, truncate=True)

        model.zero_grad()

        iteration = 0
        global_iteration = epoch * updates_epoch
        total = dataset.document_count // self.args.train_batch_size


        for batch in tqdm(sampler, total=total, desc='Train epoch %s' % epoch):
            
            model.train()
            batch = batch.to(self._device)
            # print("iteration:", global_iteration)
            if epoch < self.args.epoch_before_rel:
                # do entity detection only.
                rel_labels = None
                allow_rel = False
            else:
                rel_labels = batch.rel_labels
                allow_rel = True

            # print("current batch:", batch.encodings)


            # print("rel labels:", rel_labels)
            if self.args.model_type == 'table_filling':
                entity_labels, rel_labels = align_label(batch.entity_labels, batch.rel_labels, batch.start_token_masks)
                entity_logits, rel_logits = model(batch.encodings, batch.ctx_masks, 
                    batch.token_masks, start_labels, entity_labels, batch.entity_masks, allow_rel)
                # entity_logits = util.beam_repeat(entity_logits, self.args.beam_size)
                loss = compute_loss.compute(entity_logits, entity_labels, rel_logits, rel_labels) 
            elif self.args.model_type == 'bert_ner':
                entity_logits, rel_logits = model(batch.encodings, batch.ctx_masks)
                entity_labels = batch.entity_labels
                token_mask = batch.start_token_masks.sum(dim=1)
                loss = compute_loss.compute(entity_logits, entity_labels, token_mask) 
                           
            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self.args.train_log_iter == 0:
                self._log_train(optimizer, loss, epoch, iteration, global_iteration, dataset.label)

        return iteration

    def _eval(self, model: torch.nn.Module, compute_loss: Loss, dataset: Dataset, input_reader: JsonInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0):
        self._logger.info("Evaluate: %s" % dataset.label)

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # create evaluator
        if dataset.label == 'valid':
            evaluator = Evaluator(dataset, input_reader, self._tokenizer,
                                  self.args.model_type, self.args.example_count,
                                  self._examples_path, epoch, dataset.label, max_epoch=self.args.epochs)
        else:
            evaluator = Evaluator(dataset, input_reader, self._tokenizer,
                                  self.args.model_type, self.args.example_count,
                                  self._examples_path, epoch, dataset.label)

        # create batch sampler
        sampler = self._sampler.create_eval_sampler(dataset, self.args.eval_batch_size, 
                                                    input_reader.context_size, truncate=False)

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self.args.eval_batch_size)
            for batch in tqdm(sampler, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = batch.to(self._device)

                # run model (forward pass)
                # print(batch.ctx_masks)
            
                if self.args.model_type == 'table_filling':
                    entity_labels, rel_labels = align_label(batch.entity_labels, batch.rel_labels, batch.start_token_masks)
                    entity_clf, rel_clf = model(batch.encodings, batch.ctx_masks, batch.token_masks, 
                    input_reader._start_entity_label, entity_labels, evaluate=True) 
                    loss = compute_loss.compute(entity_clf, entity_labels, rel_clf, rel_labels, is_eval=True)  
                    entity_clf = util.beam_repeat(entity_clf, self.args.beam_size)
                    # rel_clf = util.beam_repeat(rel_clf, self.args.beam_size)
                elif self.args.model_type == 'bert_ner':
                    entity_clf, rel_clf = model(batch.encodings, batch.ctx_masks, evaluate=True) 

                    entity_labels = batch.entity_labels
                    token_mask = batch.start_token_masks.sum(dim=1)
                    loss = compute_loss.compute(entity_clf, entity_labels, token_mask, is_eval=True) 
                evaluator.eval_batch(entity_clf, rel_clf, batch, 
                                    input_reader._start_entity_label, input_reader._end_entity_label)

        global_iteration = epoch * updates_epoch + iteration
        ner_eval, rel_eval, rel_ner_eval = evaluator.compute_scores()
        
        self._log_eval(*ner_eval, *rel_eval, *rel_ner_eval, loss,
                       epoch, iteration, global_iteration, dataset.label)

        if self.args.store_examples:
            evaluator.store_examples()

        return ner_eval, rel_eval, rel_ner_eval

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self.args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss[0], global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss[0], global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss.tolist(), epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss.tolist(), epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,

                  rel_prec_micro: float, rel_rec_micro: float, rel_f1_micro: float,
                  rel_prec_macro: float, rel_rec_macro: float, rel_f1_macro: float,

                  rel_ner_prec_micro: float, rel_ner_rec_micro: float, rel_ner_f1_micro: float,
                  rel_ner_prec_macro: float, rel_ner_rec_macro: float, rel_ner_f1_macro: float,
                  loss: float, epoch: int, iteration: int, global_iteration: int, label: str):


        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_prec_micro', rel_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_micro', rel_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_micro', rel_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_prec_macro', rel_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_macro', rel_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_macro', rel_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_ner_prec_micro', rel_ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_ner_recall_micro', rel_ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_ner_f1_micro', rel_ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_ner_prec_macro', rel_ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_ner_recall_macro', rel_ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_ner_f1_macro', rel_ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'loss', loss[0], global_iteration)


        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,

                      rel_prec_micro, rel_rec_micro, rel_f1_micro,
                      rel_prec_macro, rel_rec_macro, rel_f1_macro,

                      rel_ner_prec_micro, rel_ner_rec_micro, rel_ner_f1_micro,
                      rel_ner_prec_macro, rel_ner_rec_macro, rel_ner_f1_macro,
                      loss.tolist(), epoch, iteration, global_iteration)



    def _log_datasets(self, input_reader):
        self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_labels.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        self._logger.info("Relations:")
        for r in input_reader.relation_labels.values():
            self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)
            self._logger.info("Max entity length: %s"% d.max_entity_len)

        self._logger.info("Context size: %s" % input_reader.context_size)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro',
                                                 'rel_prec_macro', 'rel_rec_macro', 'rel_f1_macro',
                                                 'rel_ner_prec_micro', 'rel_ner_rec_micro', 'rel_ner_f1_micro',
                                                 'rel_ner_prec_macro', 'rel_ner_rec_macro', 'rel_ner_f1_macro',
                                                 'loss', 'epoch', 'iteration', 'global_iteration']})
