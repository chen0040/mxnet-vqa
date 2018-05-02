import os
import sys
import pandas as pd
import logging
import numpy as np

from mxnet_vqa.data.glove import glove_word2emb_300
from mxnet_vqa.utils.text_utils import word_tokenize, pad_sequences


def get_questions(data_dir_path, max_lines_retrieved=-1, split='val'):
    if split == 'train':
        data_path = os.path.join(data_dir_path, 'data/train_qa')
    elif split == 'val':
        data_path = os.path.join(data_dir_path, 'data/val_qa')
    else:
        print('Invalid split!')
        sys.exit()

    df = pd.read_pickle(data_path)
    questions = df[['question']].values.tolist()
    if max_lines_retrieved == -1:
        return questions
    return questions[:min(max_lines_retrieved, len(questions))]


def get_questions_matrix(data_dir_path, max_lines_retrieved=-1, split='val', mode='concat'):
    if mode == 'add':
        return get_questions_matrix_sum(data_dir_path, max_lines_retrieved, split)
    else:
        return get_questions_matrix_concat(data_dir_path, max_lines_retrieved, split)


def get_questions_matrix_concat(data_dir_path, max_lines_retrieved=-1, split='val'):
    questions = get_questions(data_dir_path, max_lines_retrieved, split)
    glove_word2emb = glove_word2emb_300(data_dir_path)
    logging.debug('glove: %d words loaded', len(glove_word2emb))
    seq_list = []

    for i, question in enumerate(questions):
        words = word_tokenize(question[0])
        seq = []
        for word in words:
            emb = np.zeros(shape=300)
            if word in glove_word2emb:
                emb = glove_word2emb[word]
            seq.append(emb)
        if (i + 1) % 10000 == 0:
            logging.debug('loaded %d questions', i + 1)
        seq_list.append(seq)
    question_matrix = pad_sequences(seq_list)

    return question_matrix


def get_questions_matrix_sum(data_dir_path, max_lines_retrieved=-1, split='val'):
    questions = get_questions(data_dir_path, max_lines_retrieved, split)
    glove_word2emb = glove_word2emb_300(data_dir_path)
    logging.debug('glove: %d words loaded', len(glove_word2emb))
    seq_list = []

    for i, question in enumerate(questions):
        words = word_tokenize(question[0])
        E = np.zeros(shape=(300, len(words)))
        for j, word in enumerate(words):
            if word in glove_word2emb:
                emb = glove_word2emb[word]
                E[:, j] = emb
        E = np.sum(E, axis=1)
        if (i + 1) % 10000 == 0:
            logging.debug('loaded %d questions', i + 1)
        seq_list.append(E)
    question_matrix = pad_sequences(seq_list)

    return question_matrix