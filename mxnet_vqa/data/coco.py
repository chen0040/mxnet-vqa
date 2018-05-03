from mxnet_vqa.data.coco_answers import get_answers_matrix
from mxnet_vqa.data.coco_questions import get_questions_matrix
from mxnet_vqa.data.coco_images import get_coco_2014_val_image_features
import os
import mxnet as mx
from mxnet import nd
import numpy as np
import logging


def train_test_split(data_dir_path, question_mode='add', answer_mode='int',
                     batch_size=64, max_lines_retrieved=100000,
                     max_sequence_length=-1,
                     ctx=mx.cpu(),
                     test_size=0.2):
    answers = get_answers_matrix(data_dir_path=data_dir_path,
                                 mode=answer_mode, split='val',
                                 max_lines_retrieved=max_lines_retrieved)
    questions = get_questions_matrix(data_dir_path=data_dir_path,
                                     mode=question_mode, split='val',
                                     max_sequence_length=max_sequence_length,
                                     max_lines_retrieved=max_lines_retrieved)
    image_feats = get_coco_2014_val_image_features(data_dir_path=data_dir_path, ctx=ctx,
                                                   coco_images_dir_path=os.path.join(data_dir_path, 'val2014'),
                                                   max_lines_retrieved=max_lines_retrieved)
    record_count = answers.shape[0]
    indices = np.random.permutation(record_count)
    training_record_count = record_count - int(record_count * test_size)
    train_idx, test_idx = indices[:training_record_count], indices[training_record_count:]

    meta = dict()
    meta['questions_matrix_shape'] = questions.shape[1:]
    meta['answers_matrix_shape'] = answers.shape[1:]
    meta['img_feats_matrix_shape'] = image_feats.shape[1:]

    logging.debug('train test meta: %s', meta)

    return mx.io.NDArrayIter(data=[nd.array(image_feats[train_idx]),
                                   nd.array(questions[train_idx])],
                             label=[nd.array(answers[train_idx])], batch_size=batch_size), \
           mx.io.NDArrayIter(data=[nd.array(image_feats[test_idx]),
                                   nd.array(questions[test_idx])],
                             label=[nd.array(answers[test_idx])], batch_size=batch_size), \
           meta
