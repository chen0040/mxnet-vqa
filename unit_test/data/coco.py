import os
import pandas as pd
import sys
import unittest
import logging
import mxnet as mx


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), '..', path)


class TestLoadCocoValImages(unittest.TestCase):

    def test_load_coco_val_images(self):
        sys.path.append(patch_path('..'))
        data_dir_path = patch_path('../demo/data/coco')
        coco_image_dir_path = patch_path('../demo/data/coco/val2014')

        logging.basicConfig(level=logging.DEBUG)

        from mxnet_vqa.data.coco_images import get_coco_2014_val_images, get_coco_2014_val_image_features
        max_lines = 100000
        result = get_coco_2014_val_images(data_dir_path, coco_image_dir_path, max_lines_retrieved=max_lines)
        self.assertEqual(max_lines, len(result))
        img_feats = get_coco_2014_val_image_features(data_dir_path,
                                                     coco_image_dir_path,
                                                     max_lines_retrieved=max_lines,
                                                     ctx=mx.gpu(0))
        print('img_features: ', img_feats.shape)
        self.assertEqual(max_lines, len(img_feats))


class TestCocoValSet(unittest.TestCase):

    def test_coco_val(self):
        sys.path.append(patch_path('..'))
        data_dir_path = patch_path('../demo/data/coco')
        data_path = os.path.join(data_dir_path, 'data/val_qa')

        logging.basicConfig(level=logging.DEBUG)

        df = pd.read_pickle(data_path)
        print(df.head())

        from mxnet_vqa.data.coco_questions import get_questions_matrix
        from mxnet_vqa.data.coco_answers import get_answers_matrix

        max_lines = 100000
        expected_max_sequence_length = 25
        glove_dim = 300
        questions = get_questions_matrix(data_dir_path, max_lines_retrieved=max_lines)
        self.assertTupleEqual((max_lines, expected_max_sequence_length, glove_dim), questions.shape)
        questions = get_questions_matrix(data_dir_path, max_lines_retrieved=max_lines, mode='add')
        self.assertTupleEqual((max_lines, glove_dim), questions.shape)

        nb_classes = 1001
        answers = get_answers_matrix(data_dir_path, max_lines_retrieved=max_lines)
        self.assertTupleEqual((max_lines, nb_classes), answers.shape)
        answers = get_answers_matrix(data_dir_path, max_lines_retrieved=max_lines, mode='int')
        self.assertTupleEqual((max_lines,), answers.shape)

        # features = get_coco_features(data_dir_path)
        # print('features: ', features.shape)


class TestCocoDataIter(unittest.TestCase):

    def test_data_iter(self):
        sys.path.append(patch_path('..'))
        data_dir_path = patch_path('../demo/data/coco')

        logging.basicConfig(level=logging.DEBUG)

        from mxnet_vqa.data.coco import train_test_split

        max_lines = 100000
        train_data, test_data = train_test_split(data_dir_path=data_dir_path,
                                                 answer_mode='int',
                                                 question_mode='add',
                                                 batch_size=64,
                                                 max_lines_retrieved=max_lines)

        for i, batch in enumerate(train_data):
            if i == 10:
                break
            batch_images = batch.data[0]
            batch_questions = batch.data[1]
            batch_answers = batch.data[2]
            print('batch images: ', batch_images.shape, 'batch questions', batch_questions.shape,
                  'batch answers: ', batch_answers.shape)


if __name__ == '__main__':
    unittest.main()
