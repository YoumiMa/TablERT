import codecs
import sys, os
import json
from transformers import BertTokenizer


MAX_LENGTH = 256

from itertools import groupby


tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

def sent2doc(sent_f: str, doc_f: str):


    with codecs.open(sent_f, 'r') as f:
        sent_data = json.load(f)

    doc_data = []
    for doc_id, group in groupby(sent_data, lambda e:e['orig_id']):
        doc_data.append([g for g in group])

    trunked_doc_cnt = 1
    trunked_doc_data = []


    for data in doc_data:
        cumsum = 0
        data_unit = {'tokens': [], 'relations': [], 'tags': []}
        for i, sent in enumerate(data[1:]):
            sent_encoding = []
            for tok in sent['tokens']:
                sent_encoding.extend(tokenizer.encode(tok, add_special_tokens=False))
            cumsum += len(sent_encoding) + 2

            if cumsum + 2 > MAX_LENGTH:
                data_unit['orig_id'] = trunked_doc_cnt
                trunked_doc_cnt += 1
                trunked_doc_data.append(data_unit)
                data_unit = {'tokens': [], 'relations': [], 'tags': []}
                cumsum = len(sent_encoding) + 2

            data_unit['tokens'].extend(sent['tokens'])
            data_unit['relations'].extend(sent['relations'])
            data_unit['tags'].extend(sent['tags'])
            if i != len(data) - 2:
                data_unit['tokens'].extend(['[SEP]'])
                data_unit['tags'].extend(['O'])
        
        data_unit['orig_id'] = trunked_doc_cnt

        trunked_doc_cnt += 1
        trunked_doc_data.append(data_unit)

    with codecs.open(doc_f, "w") as w:
        json.dump(trunked_doc_data, w)


    return




if __name__ == "__main__":
    if "-h" in sys.argv[1]:
        print("python3 sent2doc.py SENTENCE_FILE DOC_FILE")
    else:
        sent2doc(sys.argv[1], sys.argv[2])