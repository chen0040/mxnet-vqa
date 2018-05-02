import sys
from collections import defaultdict
import pandas as pd
import scipy as sc
import os
import numpy as np
from mxnet_vqa.data.glove import glove_word2emb_300
from mxnet_vqa.utils.text_utils import pad_sequences


def int_to_answers(data_dir_path, split):
    data_path = os.path.join(data_dir_path, 'data/val_qa')
    if split == 'train':
        data_path = os.path.join(data_dir_path, 'data/train_qa')
    df = pd.read_pickle(data_path)
    answers = df[['multiple_choice_answer']].values.tolist()
    freq = defaultdict(int)
    for answer in answers:
        freq[answer[0].lower()] += 1
    int_to_answer = sorted(freq.items(), key=sc.operator.itemgetter(1), reverse=True)[0:1000]
    int_to_answer = [answer[0] for answer in int_to_answer]
    return int_to_answer


def answers_to_onehot(data_dir_path, split='val'):
    top_answers = int_to_answers(data_dir_path, split)
    answer_to_onehot = {}
    for i, word in enumerate(top_answers):
        onehot = np.zeros(1001)
        onehot[i] = 1.0
        answer_to_onehot[word] = onehot
    return answer_to_onehot


def get_answers_matrix(data_dir_path, split):
    if split == 'train':
        data_path = os.path.join(data_dir_path, 'data/train_qa')
    elif split == 'val':
        data_path = os.path.join(data_dir_path, 'data/val_qa')
    else:
        print('Invalid split!')
        sys.exit()

    df = pd.read_pickle(data_path)
    answers = df[['multiple_choice_answer']].values.tolist()
    answer_matrix = np.zeros((len(answers), 1001))
    default_onehot = np.zeros(1001)
    default_onehot[1000] = 1.0

    answer_to_onehot_dict = answers_to_onehot(data_dir_path, split)

    for i, answer in enumerate(answers):
        answer_matrix[i] = answer_to_onehot_dict.get(answer[0].lower(), default_onehot)

    return answer_matrix


def get_questions(data_dir_path, split='val'):
    if split == 'train':
        data_path = os.path.join(data_dir_path, 'data/train_qa')
    elif split == 'val':
        data_path = os.path.join(data_dir_path, 'data/val_qa')
    else:
        print('Invalid split!')
        sys.exit()

    df = pd.read_pickle(data_path)
    questions = df[['question']].values.tolist()
    return questions


def word_tokenize(sentence):
    return sentence.split()


def get_questions_matrix(data_dir_path, split):
    questions = get_questions(data_dir_path, split)
    glove_word2emb = glove_word2emb_300(data_dir_path)
    seq_list = []

    for question in questions:
        words = word_tokenize(question[0])
        seq = []
        for word in words:
            emb = np.zeros(shape=300)
            if word in glove_word2emb:
                emb = glove_word2emb[word]
                seq.append(emb)
        seq_list.append(seq)
    question_matrix = pad_sequences(seq_list)

    return question_matrix



