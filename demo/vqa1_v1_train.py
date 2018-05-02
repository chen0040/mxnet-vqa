import os
import sys
import logging
import mxnet as mx

def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def main():
    sys.path.append(patch_path('..'))
    data_dir_path = patch_path('data/coco')
    model_dir_path = patch_path('models')

    answer_mode = 'int'
    question_mode = 'add'
    batch_size = 64
    epochs = 10

    logging.basicConfig(level=logging.DEBUG)

    ctx = mx.gpu(0)

    from mxnet_vqa.data.coco import train_test_split
    train_data, test_data, meta_data = train_test_split(data_dir_path=data_dir_path,
                                                        max_lines_retrieved=100000,
                                                        answer_mode=answer_mode,
                                                        ctx=ctx,
                                                        question_mode=question_mode,
                                                        batch_size=batch_size)

    from mxnet_vqa.library.vqa1 import VQANet
    net = VQANet(model_ctx=ctx)
    net.input_mode_question = question_mode
    net.input_mode_answer = answer_mode
    net.version = '1'
    net.fit(data_train=train_data, data_eva=test_data, meta=meta_data,
            model_dir_path=model_dir_path, epochs=epochs)


if __name__ == '__main__':
    main()
