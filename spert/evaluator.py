import os
import warnings
from typing import List, Tuple, Dict

import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from transformers import BertTokenizer

from spert.entities import Document, Dataset, EntityLabel, EntityType
from spert.input_reader import JsonInputReader
from spert.opt import jinja2
from spert.sampling import EvalTensorBatch

from functools import reduce

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))



class Evaluator:
    def __init__(self, dataset: Dataset, input_reader: JsonInputReader, text_encoder: BertTokenizer,
                 rel_filter_threshold: float, example_count: int, example_path: str,
                 epoch: int, dataset_label: str):
        self._text_encoder = text_encoder
        self._input_reader = input_reader
        self._dataset = dataset
        self._rel_filter_threshold = rel_filter_threshold

        self._epoch = epoch
        self._dataset_label = dataset_label

        self._example_count = example_count
        self._examples_path = example_path

        # relations
        self._gt_relations = []  # ground truth
        self._pred_relations = []  # prediction

        # entities
        self._gt_entities = []  # ground truth
        self._pred_entities = []  # prediction

        self._pseudo_entity_label = EntityLabel('Entity', 1, 'Entity', 'Entity')
        self._pseudo_entity_type = EntityType([self._pseudo_entity_label], 1, 'Entity', 'Entity')  # for span only evaluation

        self._convert_gt(self._dataset.documents)

    def eval_batch(self, batch_entity_clf: List[torch.tensor], 
                    batch_rel_clf: List[torch.tensor],
                   batch: EvalTensorBatch):
        batch_size = len(batch_entity_clf)
        
        for i in range(batch_size):
            # get model predictions for sample
            entity_clf = batch_entity_clf[i]
            rel_clf = batch_rel_clf[i]

            entity_scores, entity_preds = torch.max(entity_clf, dim=2)

            entity_preds = torch.ceil(entity_preds.float() / 4)
            pred_entities = self._convert_pred_entities_(entity_preds.squeeze(0), entity_scores.squeeze(0))
            self._pred_entities.append(pred_entities)

            rel_scores, rel_preds = torch.max(rel_clf, dim=2)

            # print("rel_preds:", rel_preds)
            rel_preds_split = rel_preds.squeeze(0).split([i for i in range(entity_preds.shape[-1]-1, 0, -1)],dim=0)

            pred_relations = self._convert_pred_relations(rel_preds_split, rel_scores.squeeze(0), 
                                                            pred_entities)
            self._pred_relations.append(pred_relations)    


    def compute_scores(self):
        print("Evaluation")

        print("")
        print("--- Entities (NER) ---")
        print("")
        merged_gt = self._merge(self._gt_entities)
        merged_pred = self._merge(self._pred_entities)
        gt, pred = self._convert_by_setting(merged_gt, merged_pred, include_entity_types=True)
        ner_eval = self._score(gt, pred, print_results=True)

        print("")
        print("--- Relations ---")
        print("")
        print("Without NER")
        # print("gt relations:", self._gt_relations)
        gt, pred = self._convert_by_setting(self._gt_relations, self._pred_relations, include_entity_types=False)
        # print("gt:", [r[0][2].verbose_name for r in gt])
        # print("pred:", [r[0][2].verbose_name for r in pred])
        rel_eval = self._score(gt, pred, print_results=True)

        print("")
        print("With NER")
        gt, pred = self._convert_by_setting(self._gt_relations, self._pred_relations, include_entity_types=True)
        rel_ner_eval = self._score(gt, pred, print_results=True)

        return ner_eval, rel_eval, rel_ner_eval

    def store_examples(self):
        if jinja2 is None:
            warnings.warn("Examples cannot be stored since Jinja2 is not installed.")
            return

        entity_examples = []
        rel_examples = []
        rel_examples_ner = []

        for i, doc in enumerate(self._dataset.documents):
            # entities
            entity_example = self._convert_example(doc, self._gt_entities[i], self._pred_entities[i],
                                                   include_entity_types=True, to_html=self._entity_to_html)
            entity_examples.append(entity_example)

            # relations
            # without entity types
            rel_example = self._convert_example(doc, self._gt_relations[i], self._pred_relations[i],
                                                include_entity_types=False, to_html=self._rel_to_html)
            rel_examples.append(rel_example)

            # with entity types
            rel_example_ner = self._convert_example(doc, self._gt_relations[i], self._pred_relations[i],
                                                    include_entity_types=True, to_html=self._rel_to_html)
            rel_examples_ner.append(rel_example_ner)

        label, epoch = self._dataset_label, self._epoch

        with open("text", 'w') as f:
            text = [r['text'] for r in rel_examples]
            f.writelines(text)
        # entities
        self._store_examples(entity_examples[:self._example_count],
                             file_path=self._examples_path % ('entities', label, epoch),
                             template='entity_examples.html')

        self._store_examples(sorted(entity_examples[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % ('entities_sorted', label, epoch),
                             template='entity_examples.html')

        # relations
        # without entity types
        self._store_examples(rel_examples[:self._example_count],
                             file_path=self._examples_path % ('rel', label, epoch),
                             template='relation_examples.html')

        self._store_examples(sorted(rel_examples[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % ('rel_sorted', label, epoch),
                             template='relation_examples.html')

        # with entity types
        self._store_examples(rel_examples_ner[:self._example_count],
                             file_path=self._examples_path % ('rel_ner', label, epoch),
                             template='relation_examples.html')

        self._store_examples(sorted(rel_examples_ner[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % ('rel_ner_sorted', label, epoch),
                             template='relation_examples.html')

    def _convert_gt(self, docs: List[Document]):
        for doc in docs:
            gt_relations = doc.relations
            gt_entities = doc.entities

            # convert ground truth relations and entities for precision/recall/f1 evaluation
            sample_gt_relations = [rel.as_tuple() for rel in gt_relations]
            sample_gt_entities = [entity.as_tuple() for entity in gt_entities]
            # print(sample_gt_relations)
            self._gt_relations.append(sample_gt_relations)
            self._gt_entities.append(sample_gt_entities)

    def _convert_pred_entities(self, pred_types: torch.tensor, pred_scores: torch.tensor):
        converted_preds = []

        for i in range(pred_types.shape[0]):
            label_idx = pred_types[i].item()
            entity_type = self._input_reader.get_entity_type(label_idx)

            start, end = pred_spans[i].tolist()
            score = pred_scores[i].item()

            converted_pred = (start, end, entity_type, score)
            converted_preds.append(converted_pred)

        return converted_preds


    def _merge(self, entities):
        for i in range(len(entities)):
            to_add = []
            for j in range(len(entities[i])-1):
                if entities[i][j][1] == entities[i][j+1][0] and entities[i][j][2] == entities[i][j+1][2]:
                    to_add.append((entities[i][j][0], entities[i][j+1][1], entities[i][j][2]))
                    j += 1
                else:
                    to_add.append(entities[i][j])

        return entities

    def _convert_pred_entities_(self, pred_types: torch.tensor, pred_scores: torch.tensor):
        converted_preds = []
        # print(pred_types)
        
        curr_type = 0
        start = 0

        for i in range(pred_types.shape[0]):
            type_idx = pred_types[i].item()
            # print("entity type:", curr_type)
            score = pred_scores[i].item()
            if type_idx != curr_type:
                if curr_type != 0:
                    converted_pred = (start+1, i+1, entity_type, score)
                    # print("appended:", start+1, i+1, entity_type.index)
                    converted_preds.append(converted_pred)
                start = i
                curr_type = type_idx
                entity_type = self._input_reader.get_entity_type(curr_type)
        if curr_type != 0:
            converted_pred = (start+1, i+2 , entity_type, score)
            # print("appended:", start+1, i+2, entity_type.index)
            converted_preds.append(converted_pred)            

        # print('???????',[preds[2].index for preds in converted_preds])
        return converted_preds

    def _convert_pred_relations(self, pred_types: List[torch.tensor], pred_scores: torch.tensor, 
                                pred_entities: List[tuple]):
        converted_rels = []
        # print(pred_types)
        for i in range(len(pred_types)):
            for j in range(pred_types[i].shape[-1]):
                label_idx = pred_types[i][j].float()
                if label_idx != 0:    
                    pred_rel_type = self._input_reader.get_relation_type(torch.ceil(label_idx/2).item())
                    if label_idx in self._input_reader._right_rel_label: # R-X
                        head_idx = i 
                        tail_idx = i + j + 1
                    else: # L-X
                        head_idx = i + j + 1
                        tail_idx = i
                    # print(head_idx, tail_idx)
                    head_entity = self._find_entity(head_idx, pred_entities)
                    # print("head entity:", head_entity)
                    tail_entity = self._find_entity(tail_idx, pred_entities)
                    # print("tail entity:", tail_entity)
                    if head_entity == None or tail_entity == None:
                        continue
                    pred_head_type = head_entity[2]
                    pred_tail_type = tail_entity[2]
                    score = pred_scores[i].item()

                    head_start, head_end = head_entity[0], head_entity[1]
                    tail_start, tail_end = tail_entity[0], tail_entity[1]
                    converted_rel = ((head_start, head_end, pred_head_type),
                                     (tail_start, tail_end, pred_tail_type), pred_rel_type, score)

                    converted_rels.append(converted_rel)
        # print(converted_rels)
        return converted_rels

    def _find_entity(self, idx, entities):
        for e in entities:
            if e[0]-1 <= idx < e[1]-1 :
                return e
        return None

    def _convert_by_setting(self, gt: List[List[Tuple]], pred: List[List[Tuple]],
                            include_entity_types: bool = True, include_score: bool = False):
        # either include or remove entity types based on setting
        def convert(t):
            if not include_entity_types:
                # remove entity type and score for evaluation
                if type(t[0]) == int:  # entity
                    c = [t[0], t[1], self._pseudo_entity_type]
                else:  # relation
                    c = [(t[0][0], t[0][1], self._pseudo_entity_type),
                         (t[1][0], t[1][1], self._pseudo_entity_type), t[2]]
            else:
                c = list(t[:3])

            if include_score and len(t) > 3:
                # include prediction scores
                c.append(t[3])

            return tuple(c)

        converted_gt, converted_pred = [], []

        for sample_gt, sample_pred in zip(gt, pred):

            converted_gt.append([convert(t) for t in sample_gt])
            converted_pred.append([convert(t) for t in sample_pred])
        return converted_gt, converted_pred

    def _score(self, gt: List[List[Tuple]], pred: List[List[Tuple]], print_results: bool = False):
        assert len(gt) == len(pred)

        gt_flat = []
        pred_flat = []
        types = set()


        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()
            union.update(sample_gt)
            union.update(sample_pred)

            for s in union:
                if s in sample_gt:
                    t = s[2]
                    gt_flat.append(t.index)
                    types.add(t)
                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    t = s[2]
                    pred_flat.append(t.index)
                    types.add(t)
                else:
                    pred_flat.append(0)

        # print("gt_flat:", gt_flat)
        # print("pred_flat:", pred_flat)
        # print("types:", [t.short_name for t in types])
        metrics = self._compute_metrics(gt_flat, pred_flat, types, print_results)
        return metrics

    def _compute_metrics(self, gt_all, pred_all, types, print_results: bool = False):
        labels = [t.index for t in types]
        gt_all = [gt for gt in gt_all]
        pred_all = [pred for pred in pred_all]
        per_type = prfs(gt_all, pred_all, labels=labels, average=None)
        micro = prfs(gt_all, pred_all, labels=labels, average='micro')[:-1]
        macro = prfs(gt_all, pred_all, labels=labels, average='macro')[:-1]
        total_support = sum(per_type[-1])

        if print_results:
            self._print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

        return [m * 100 for m in micro + macro]

    def _print_results(self, per_type: List, micro: List, macro: List, types: List):
        columns = ('type', 'precision', 'recall', 'f1-score', 'support')

        row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
        results = [row_fmt % columns, '\n']

        metrics_per_type = []
        for i, t in enumerate(types):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, types):
            results.append(row_fmt % self._get_row(m, t.short_name))
            results.append('\n')

        results.append('\n')

        # micro
        results.append(row_fmt % self._get_row(micro, 'micro'))
        results.append('\n')

        # macro
        results.append(row_fmt % self._get_row(macro, 'macro'))

        results_str = ''.join(results)
        print(results_str)

    def _get_row(self, data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)

    def _convert_example(self, doc: Document, gt: List[Tuple], pred: List[Tuple],
                         include_entity_types: bool, to_html):
        encoding = doc.encoding

        gt, pred = self._convert_by_setting([gt], [pred], include_entity_types=include_entity_types, include_score=True)

        gt, pred = gt[0], pred[0]

        # get micro precision/recall/f1 scores
        if gt or pred:
            pred_s = [p[:3] for p in pred]  # remove score
            precision, recall, f1 = self._score([gt], [pred_s])[:3]
        else:
            # corner case: no ground truth and no predictions
            precision, recall, f1 = [100] * 3

        scores = [p[-1] for p in pred]
        # pred = [p[:-1] for p in pred]
        union = set(gt + pred)

        # true positives
        tp = []
        # false negatives
        fn = []
        # false positives
        fp = []

        for s in union:
            type_verbose = s[2].verbose_name
            # print("vvvv:", type_verbose)

            if s in gt:
                if s in pred:
                    score = scores[pred.index(s)]
                    tp.append((to_html(s, encoding), type_verbose, score))
                else:
                    fn.append((to_html(s, encoding), type_verbose, -1))
            else:
                score = scores[pred.index(s)]
                fp.append((to_html(s, encoding), type_verbose, score))

        tp = sorted(tp, key=lambda p: p[-1], reverse=True)
        fp = sorted(fp, key=lambda p: p[-1], reverse=True)

        text = self._prettify(self._text_encoder.decode(encoding))
        # print("ttttpppp:", tp)
        return dict(text=text, tp=tp, fn=fn, fp=fp, precision=precision, recall=recall, f1=f1, length=len(doc.tokens))

    def _entity_to_html(self, entity: Tuple, encoding: List[int]):
        start, end = entity[:2]
        entity_type = entity[2].verbose_name

        tag_start = ' <span class="entity">'
        tag_start += '<span class="type">%s</span>' % entity_type

        ctx_before = self._text_encoder.decode(encoding[:start])
        e1 = self._text_encoder.decode(encoding[start:end])
        ctx_after = self._text_encoder.decode(encoding[end:])

        html = ctx_before + tag_start + e1 + '</span> ' + ctx_after
        html = self._prettify(html)

        return html

    def _rel_to_html(self, relation: Tuple, encoding: List[int]):
        head, tail = relation[:2]
        head_tag = ' <span class="head"><span class="type">%s</span>'
        tail_tag = ' <span class="tail"><span class="type">%s</span>'

        if head[0] < tail[0]:
            e1, e2 = head, tail
            e1_tag, e2_tag = head_tag % head[2].verbose_name, tail_tag % tail[2].verbose_name
        else:
            e1, e2 = tail, head
            e1_tag, e2_tag = tail_tag % tail[2].verbose_name, head_tag % head[2].verbose_name

        segments = [encoding[:e1[0]], encoding[e1[0]:e1[1]], encoding[e1[1]:e2[0]],
                    encoding[e2[0]:e2[1]], encoding[e2[1]:]]

        ctx_before = self._text_encoder.decode(segments[0])
        e1 = self._text_encoder.decode(segments[1])
        ctx_between = self._text_encoder.decode(segments[2])
        e2 = self._text_encoder.decode(segments[3])
        ctx_after = self._text_encoder.decode(segments[4])

        html = (ctx_before + e1_tag + e1 + '</span> '
                     + ctx_between + e2_tag + e2 + '</span> ' + ctx_after)
        html = self._prettify(html)

        return html

    def _prettify(self, text: str):
        text = text.replace('_start_', '').replace('_classify_', '').replace('<unk>', '').replace('⁇', '')
        text = text.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
        return text

    def _store_examples(self, examples: List[Dict], file_path: str, template: str):
        template_path = os.path.join(SCRIPT_PATH, 'templates', template)

        # read template
        with open(os.path.join(SCRIPT_PATH, template_path)) as f:
            template = jinja2.Template(f.read())

        # print("example:", examples)
        # write to disc
        template.stream(examples=examples).dump(file_path)
