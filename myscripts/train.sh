CUDA_VISIBLE_DEVICES=5 nohup python -W ignore:semaphore_tracker:UserWarning ./spert.py train --config configs/train_bioul_conll03_base.conf > conll03_bert_doc_5runs&
ps -f | egrep "$$|PID"