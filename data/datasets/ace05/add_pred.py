import codecs
import sys, os
import json



def add_pred(ori, pred, output):


    with codecs.open(ori, 'r') as f:
        ori_data = json.load(f)

    with codecs.open(pred, 'r') as f:
        pred_data = json.load(f)


    for i, sent in enumerate(ori_data):
        if sent['orig_id'] == pred_data[i]['orig_id']:
            sent.update(pred_data[i])


    with codecs.open(output, 'w') as f:
        json.dump(ori_data, f)

    return

if __name__ == "__main__":
    if "-h" in sys.argv[1]:
        print("python3 add_pred.py ORIGINAL PRED OUTPUT")
    else:
        add_pred(sys.argv[1], sys.argv[2], sys.argv[3])