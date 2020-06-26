# Named Entity Recognition and Relation Extraction using Enhanced Table Filling by Contextualized Representations
This is the PyTorch code for conference submission 'Named Entity Recognition and Relation Extraction using Enhanced Table Filling by Contextualized Representations'.
The general framework of this software adopts that of [SpERT](https://github.com/markus-eberts/spert) [1].
# Setup

## Requirements
Requirments are listed in `requirements.txt`, same as SpERT [1].

Required
- Python 3.5+
- PyTorch 1.1.0+ (tested with version 1.3.1)
- transformers 2.2.0+ (tested with version 2.2.0)
- scikit-learn (tested with version 0.21.3)
- tqdm (tested with version 4.19.5)
- numpy (tested with version 1.17.4)

Optional
- jinja2 (tested with version 2.10.3) - if installed, used to export relation extraction examples
- tensorboardX (tested with version 1.6) - if installed, used to save training process to tensorboard


# Examples

We provide processed CoNLL04 [2] datasets along with the software in folder `data/datasets/`. We follow the data splits of [1] and [3].

## Training

To train a model on CoNLL04 train set and evaluate on CoNLL04 development set, run

```
python ./table_filling.py train --config configs_example/train_conll04.conf
```
## Evaluation

To evalute a model on CoNLL04 test set, fill in the field `model_path` in `configs_example/eval_conll04.conf` with the directory of the model and run

```
python ./table_filling.py eval --config configs_example/eval_conll04.conf
```

# References
```
[1]Markus Eberts and Adrian Ulges. 2020.  Span-based joint entity and relation extraction with transformerpre-training. In 24th European Conference on Artifi-cial Intelligence (ECAI).
[2]Dan Roth and Wen-tau Yih, ‘A Linear Programming Formulation forGlobal Inference in Natural Language Tasks’, in Proc. of CoNLL 2004 at HLT-NAACL 2004, pp. 1–8, Boston, Massachusetts, USA, (May 6 -May 7 2004). ACL.
[3]Pankaj Gupta,  Hinrich Schütze, and Bernt Andrassy, ‘Table Filling Multi-Task Recurrent  Neural  Network  for  Joint  Entity  and  Relation Extraction’, in Proc. of COLING 2016, pp. 2537–2547, Osaka, Japan, (December 2016). The COLING 2016 Organizing Committee.
```
