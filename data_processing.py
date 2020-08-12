import codecs
import sys, os
import json
import pickle
from typing import Dict, List


def get_orig_dataset(file: str) -> List[Dict]:

    with codecs.open(file, "r") as f:
        data = json.load(f)

    return data 


def get_bioul_dataset(file: str) -> Dict[int, List[str]]:

    objects = []

    # dump all infos into objects.
    with codecs.open(file, "rb") as f:
        while True:
            try:
                objects.append(pickle.load(f))
            except EOFError:
                break

    contents = objects[3]
    contents.update(objects[8])
    for c in contents:
        contents[c] = contents[c].split()
    # print(contents)
    # for idx in bioul_idx:

    return contents


def update_ner(orig: List[Dict] , bioul: Dict[int, List[str]]) -> List[Dict]:

    for data in orig:
        if data["orig_id"] in bioul:
            data["tags"] = bioul[data["orig_id"]]
        data.pop("entities")
    return orig


def update_rel(dataset: List[Dict]) -> List[Dict]:

    for data in dataset:
        entities = data["entities"]
        for r in data["relations"]:
            r["head"] = entities[r["head"]]["end"] - 1
            r["tail"] = entities[r["tail"]]["end"] - 1
        data.pop("entities")

    return dataset



def update_dataset(file: str, bioul_file: str) -> None:

    orig_data = get_orig_dataset(file)
    bioul_data = get_bioul_dataset(bioul_file)
    new_data = update_ner(orig_data, bioul_data)
    # new_data = update_rel(new_data)
    
    target_dir = os.path.split(file)[0] + '_bioul'

    os.makedirs(target_dir, exist_ok=True)

    with codecs.open(os.path.join(target_dir, os.path.split(file)[1]), "w") as w:
        json.dump(new_data, w)
        
    return


if __name__ == "__main__":
    if "-h" in sys.argv[1]:
        print("python3 data_processing.py FILE BIOUL_FILE")
    else:
        update_dataset(sys.argv[1], sys.argv[2])