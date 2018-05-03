import os
import sys
import logging
import mxnet as mx
from random import shuffle


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def main():
    sys.path.append(patch_path('..'))
    data_dir_path = patch_path('data/coco')
    model_dir_path = patch_path('models')

    max_lines = 1000

    logging.basicConfig(level=logging.DEBUG)

    ctx = mx.gpu(0)

    from mxnet_vqa.data.coco_images import get_coco_2014_val_images
    from mxnet_vqa.data.coco_answers import int_to_answers, get_answers
    from mxnet_vqa.data.coco_questions import get_questions

    answer_labels = int_to_answers(data_dir_path)

    images = get_coco_2014_val_images(data_dir_path, coco_images_dir_path=os.path.join(data_dir_path, 'val2014'),
                                      max_lines_retrieved=max_lines)
    questions = get_questions(data_dir_path, max_lines_retrieved=max_lines)
    answers = get_answers(data_dir_path, max_lines_retrieved=max_lines)

    from mxnet_vqa.library.vqa1 import VQANet
    net = VQANet(model_ctx=ctx)
    net.version = '2'
    net.load_glove_300(data_dir_path)
    net.load_model(model_dir_path)

    testing_data = list()
    for image_path, question, answer in zip(images, questions, answers):
        testing_data.append((image_path, question[0], answer[0]))

    shuffle(testing_data)

    for image_path, question, answer in testing_data[:30]:
        ans_class = net.predict_answer_class(image_path, question)
        if ans_class < len(answer_labels):
            predicted_answer = answer_labels[ans_class]
        else:
            predicted_answer = 'unknown'
        print('image: ', image_path)
        print('question is: ', question)
        print('predicted: ', predicted_answer, 'actual: ', answer)


if __name__ == '__main__':
    main()
