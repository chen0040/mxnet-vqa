from mxnet_vqa.data.coco_answers import get_answers_matrix
from mxnet_vqa.data.coco_questions import get_questions_matrix
from mxnet_vqa.data.coco_images import get_coco_2014_val_image_features
import os
import mxnet as mx
from mxnet import nd
import numpy as np


def train_test_split(data_dir_path, question_mode='add', answer_mode='int',
                     batch_size=64, max_lines_retrieved=100000,
                     test_size=0.2):
    answers = get_answers_matrix(data_dir_path=data_dir_path,
                                 mode=answer_mode, split='val',
                                 max_lines_retrieved=max_lines_retrieved)
    questions = get_questions_matrix(data_dir_path=data_dir_path,
                                     mode=question_mode, split='val',
                                     max_lines_retrieved=max_lines_retrieved)
    image_feats = get_coco_2014_val_image_features(data_dir_path=data_dir_path,
                                                   coco_images_dir_path=os.path.join(data_dir_path, 'val2014'),
                                                   max_lines_retrieved=max_lines_retrieved)
    record_count = answers.shape[0]
    indices = np.random.permutation(record_count)
    training_record_count = record_count - int(record_count * test_size)
    train_idx, test_idx = indices[:training_record_count], indices[training_record_count:]
    return mx.io.NDArrayIter(data=[nd.array(image_feats[train_idx]),
                                   nd.array(questions[train_idx]),
                                   nd.array(answers[train_idx])], batch_size=batch_size), \
           mx.io.NDArrayIter(data=[nd.array(image_feats[test_idx]),
                                   nd.array(questions[test_idx]),
                                   nd.array(answers[test_idx])], batch_size=batch_size)
