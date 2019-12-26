import csv
import json
import os
import random
import shutil

import numpy as np
import torch

CSV_DELIMETER = ';'


def create_directories_file(f):
    d = os.path.dirname(f)

    if d and not os.path.exists(d):
        os.makedirs(d)

    return f


def create_directories_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d)

    return d


def create_csv(file_path, *column_names):
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)

            if column_names:
                writer.writerow(column_names)


def append_csv(file_path, *row):
    if not os.path.exists(file_path):
        raise Exception("The given file doesn't exist")

    with open(file_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row)


def append_csv_multiple(file_path, *rows):
    if not os.path.exists(file_path):
        raise Exception("The given file doesn't exist")

    with open(file_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            writer.writerow(row)


def read_csv(file_path):
    lines = []
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            lines.append(row)

    return lines[0], lines[1:]


def copy_python_directory(source, dest, ignore_dirs=None):
    source = source if source.endswith('/') else source + '/'
    for (dir_path, dir_names, file_names) in os.walk(source):
        tail = '/'.join(dir_path.split(source)[1:])
        new_dir = os.path.join(dest, tail)

        if ignore_dirs and True in [(ignore_dir in tail) for ignore_dir in ignore_dirs]:
            continue

        create_directories_dir(new_dir)

        for file_name in file_names:
            if file_name.endswith('.py'):
                file_path = os.path.join(dir_path, file_name)
                shutil.copy2(file_path, new_dir)


def save_dict(log_path, dic, name):
    # save arguments
    # 1. as json
    path = os.path.join(log_path, '%s.json' % name)
    f = open(path, 'w')
    json.dump(vars(dic), f)
    f.close()

    # 2. as string
    path = os.path.join(log_path, '%s.txt' % name)
    f = open(path, 'w')
    args_str = ["%s = %s" % (key, value) for key, value in vars(dic).items()]
    f.write('\n'.join(args_str))
    f.close()


def summarize_dict(summary_writer, dic, name):
    table = 'Argument|Value\n-|-'

    for k, v in vars(dic).items():
        row = '\n%s|%s' % (k, v)
        table += row
    summary_writer.add_text(name, table)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_logger(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    for f in logger.filters[:]:
        logger.removeFilters(f)


def flatten(l):
    return [i for p in l for i in p]


def get_as_list(dic, key):
    if key in dic:
        return [dic[key]]
    else:
        return []


def extend_tensor(tensor, c, dim=0, fill=0):
    shape = list(tensor.shape)
    shape[dim] = c
    extension = torch.zeros(shape, dtype=tensor.dtype).to(tensor.device)
    extension = extension.fill_(fill)
    extended_tensor = torch.cat([tensor, extension], dim=dim)
    return extended_tensor


def padded_stack(tensors, padding=0):
    max_size = max([t.shape[0] for t in tensors])
    padded_tensors = []

    for t in tensors:
        s = max_size - t.shape[0]
        e = extend_tensor(t, s, dim=0, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked


def batch_index(tensor, index, pad=False):
    if tensor.shape[0] != index.shape[0]:
        raise Exception()

    if not pad:
        return torch.stack([tensor[i][index[i]] for i in range(index.shape[0])])
    else:
        return padded_stack([tensor[i][index[i]] for i in range(index.shape[0])])


def padded_nonzero(tensor, padding=0):
    indices = padded_stack([tensor[i].nonzero().view(-1) for i in range(tensor.shape[0])], padding)
    return indices


def swap(v1, v2):
    return v2, v1
