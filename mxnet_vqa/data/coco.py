from mxnet_vqa.data.coco_answers import get_answers_matrix
from mxnet_vqa.data.coco_questions import get_questions_matrix
from mxnet_vqa.data.coco_images import get_coco_2014_val_image_features
import os
import mxnet as mx

def load_data_iterator(data_dir_path, question_mode='add', answer_mode='int', max_lines_retrieved=100000):
    answers = get_answers_matrix(data_dir_path=data_dir_path,
                                 mode=answer_mode, split='val',
                                 max_lines_retrieved=max_lines_retrieved)
    questions = get_questions_matrix(data_dir_path=data_dir_path,
                                     mode=question_mode, split='val',
                                     max_lines_retrieved=max_lines_retrieved)
    image_feats = get_coco_2014_val_image_features(data_dir_path=data_dir_path,
                                                   coco_images_dir_path=os.path.join(data_dir_path, 'val2014'),
                                                   max_lines_retrieved=max_lines_retrieved)
