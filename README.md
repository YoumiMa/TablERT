# TablERT: Named Entity Recognition and Relation Extraction using Enhanced Table Filling by Contextualized Representations
This is the PyTorch code for the preprint ['Named Entity Recognition and Relation Extraction using Enhanced Table Filling by Contextualized Representations'](https://arxiv.org/abs/2010.07522) (TablERT). An extended version of this paper is accepted as a journal paper at [JNLP](https://www.jstage.jst.go.jp/article/jnlp/29/1/29_187/_article/-char/ja).
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

We assume the processed CoNLL04 [2] data are already located in the folder `data/datasets/`, following the data split of [1] and [3]. (not provided along with this software)

To obtain the data, it might be convenient to use our pre-processing script, i.e, data_processing.py. To envoke the script, json files provided by [SpERT](https://github.com/markus-eberts/spert), and files recording the BILOU-format annotations for the dataset provided by another project [GlobalNormalization](http://cistern.cis.lmu.de/globalNormalization/) should be used. 

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

# Citation

If you use the provided code in your work, please cite the following paper:

```
@article{Youmi Ma2022,
  title={Named Entity Recognition and Relation Extraction Using Enhanced Table Filling by Contextualized Representations},
  author={Youmi Ma and Tatsuya Hiraoka and Naoaki Okazaki},
  journal={Journal of Natural Language Processing},
  volume={29},
  number={1},
  pages={187-223},
  year={2022},
  doi={10.5715/jnlp.29.187}
}
```

# References
```
[1]Markus Eberts and Adrian Ulges, 2020, 'Span-based joint entity and relation extraction with transformerpre-training' In 24th European Conference on Artifi-cial Intelligence (ECAI).
[2]Dan Roth and Wen-tau Yih, 2004, ‘A Linear Programming Formulation forGlobal Inference in Natural Language Tasks’, in Proc. of CoNLL 2004 at HLT-NAACL 2004, pp. 1–8.
[3]Pankaj Gupta, Hinrich Schütze, and Bernt Andrassy, 2016, ‘Table Filling Multi-Task Recurrent  Neural  Network  for  Joint  Entity  and  Relation Extraction’, in Proc. of COLING 2016, pp. 2537–2547.
[4]Heike Adel and Hinrich Schütze, 2017, 'Global Normalization of Convolutional Neural Networks for Joint Entity and Relation Classification', EMNLP 2017, pp. 1723--1729. 
```


