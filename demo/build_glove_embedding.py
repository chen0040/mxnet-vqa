import os
import sys
import logging


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def main():
    sys.path.append(patch_path('..'))
    data_dir_path = patch_path('data/coco')

    logging.basicConfig(level=logging.DEBUG)
    from mxnet_vqa.data.glove import glove_word2emb_300

    emb = glove_word2emb_300(data_dir_path)
    print('glove embedding: ', len(emb))


if __name__ == '__main__':
    main()
