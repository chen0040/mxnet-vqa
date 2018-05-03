import os
import sys
import logging
import mxnet as mx


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def main():
    sys.path.append(patch_path('..'))
    data_dir_path = patch_path('data/coco')
    coco_image_dir_path = patch_path('data/coco/val2014')

    logging.basicConfig(level=logging.DEBUG)

    from mxnet_vqa.data.coco_images import get_coco_2014_val_images, get_coco_2014_val_image_features
    max_lines = 100000
    result = get_coco_2014_val_images(data_dir_path, coco_image_dir_path, max_lines_retrieved=max_lines)
    print('val_images: ', result.shape)
    img_feats = get_coco_2014_val_image_features(data_dir_path,
                                                 coco_image_dir_path,
                                                 max_lines_retrieved=max_lines,
                                                 ctx=mx.gpu(0))
    print('img_features: ', img_feats.shape)


if __name__ == '__main__':
    main()