import logging
from collections import Counter
import os
import pandas as pd
import sys
import numpy as np

def int_to_answers(data_dir_path, split):
    data_path = os.path.join(data_dir_path, 'data/val_qa')
    if split == 'train':
        data_path = os.path.join(data_dir_path, 'data/train_qa')
    df = pd.read_pickle(data_path)
    answers = df[['multiple_choice_answer']].values.tolist()
    freq = Counter()
    for answer in answers:
        freq[answer[0].lower()] += 1
    max_vocab_size = 1000
    int_to_answer = list()
    for idx, word in enumerate(freq.most_common(max_vocab_size)):
        int_to_answer.append(word[0])
    return int_to_answer


def answers_to_onehot(data_dir_path, split='val'):
    top_answers = int_to_answers(data_dir_path, split)
    answer_to_onehot = {}
    for i, word in enumerate(top_answers):
        onehot = np.zeros(1001)
        onehot[i] = 1.0
        answer_to_onehot[word] = onehot
    return answer_to_onehot


def answers_to_int(data_dir_path, split='val'):
    top_answers = int_to_answers(data_dir_path, split)
    return dict([(i, word) for i, word in enumerate(top_answers)])


def get_answers_matrix(data_dir_path, max_lines_retrieved=-1, split='val', mode='one_hot'):
    if split == 'train':
        data_path = os.path.join(data_dir_path, 'data/train_qa')
    elif split == 'val':
        data_path = os.path.join(data_dir_path, 'data/val_qa')
    else:
        print('Invalid split!')
        sys.exit()

    df = pd.read_pickle(data_path)
    answers = df[['multiple_choice_answer']].values.tolist()

    if max_lines_retrieved != -1:
        answers = answers[:min(max_lines_retrieved, len(answers))]

    if mode == 'one_hot':
        answer_matrix = np.zeros((len(answers), 1001))
        default_onehot = np.zeros(1001)
        default_onehot[1000] = 1.0

        answer_to_onehot_dict = answers_to_onehot(data_dir_path, split)

        for i, answer in enumerate(answers):
            answer_matrix[i] = answer_to_onehot_dict.get(answer[0].lower(), default_onehot)
            if (i + 1) % 10000 == 0:
                logging.debug('loaded %d answers', i + 1)

        return answer_matrix
    elif mode == 'int':
        answer_matrix = np.zeros((len(answers)))

        answers_to_int_dict = answers_to_int(data_dir_path, split)

        default_val = 1000

        for i, answer in enumerate(answers):
            answer_matrix[i] = answers_to_int_dict.get(answer[0].lower(), default_val)
            if (i + 1) % 10000 == 0:
                logging.debug('loaded %d answers', i + 1)
        return answer_matrix
    else:
        print('Invalid mode!')
        sys.exit()