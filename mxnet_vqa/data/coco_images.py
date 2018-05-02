import sys
import pandas as pd
from mxnet_vqa.utils.image_utils import Vgg16FeatureExtractor
import os
import numpy as np
import logging
import time
import pickle
import mxnet as mx


def load_coco_2014_val_images_dict(coco_images_dir_path):
    result = dict()
    if not os.path.exists(coco_images_dir_path):
        logging.error('Please download and extract val2014 images to %s before continuing',
                      coco_images_dir_path)
        sys.exit()

    for root, dirs, files in os.walk(coco_images_dir_path):
        for fn in files:
            if fn.lower().endswith('.jpg'):
                fp = os.path.join(root, fn)
                image_id = int(fn.replace('COCO_val2014_', '').replace('.jpg', ''))
                result[image_id] = fp
    return result


def get_coco_2014_val_images(data_dir_path, coco_images_dir_path, max_lines_retrieved=-1):
    data_path = os.path.join(data_dir_path, 'data/val_qa')
    coco_image_paths = load_coco_2014_val_images_dict(coco_images_dir_path)

    df = pd.read_pickle(data_path)
    image_id_list = df[['image_id']].values.tolist()
    result = list()
    for image_id in image_id_list:
        if image_id[0] in coco_image_paths:
            result.append(coco_image_paths[image_id[0]])
        else:
            logging.warning('Failed to get image path for image id %s', image_id[0])
        if max_lines_retrieved != -1 and len(result) == max_lines_retrieved:
            break
    return result


def checkpoint_features(pickle_path, features):
    with open(pickle_path, 'wb') as handle:
        logging.debug('saving image features as %s', pickle_path)
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_coco_2014_val_image_features(data_dir_path, coco_images_dir_path, ctx=mx.cpu(), max_lines_retrieved=-1):
    pickle_name = 'coco_val2014_feats'
    if max_lines_retrieved == -1:
        pickle_name = pickle_name + '.pickle'
    else:
        pickle_name = pickle_name + '_' + str(max_lines_retrieved) + '.pickle'
    pickle_path = os.path.join(data_dir_path, pickle_name)

    features = dict()
    if os.path.exists(pickle_path):
        logging.info('loading val2014 image features from %s', pickle_path)
        start_time = time.time()
        with open(pickle_path, 'rb') as handle:
            features = pickle.load(handle)
            duration = time.time() - start_time
            logging.debug('loading val2014 features from pickle took %.1f seconds', duration)

    fe = Vgg16FeatureExtractor(model_ctx=ctx)

    data_path = os.path.join(data_dir_path, 'data/val_qa')
    coco_image_paths = None

    df = pd.read_pickle(data_path)
    image_id_list = df[['image_id']].values.tolist()
    result = list()

    start_extracting_time = time.time()
    features_updated = False
    for i, image_id in enumerate(image_id_list):
        if image_id[0] not in features:
            if coco_image_paths is None:
                load_coco_2014_val_images_dict(coco_images_dir_path)

            if image_id[0] in coco_image_paths:
                img_path = coco_image_paths[image_id[0]]
                f = fe.extract_image_features(img_path)
                feat = f.asnumpy()
                features[image_id[0]] = feat
                features_updated = True
            else:
                logging.warning('Failed to extract image features for image id %s', image_id[0])
        if max_lines_retrieved != -1 and (i + 1) == max_lines_retrieved:
            break
        if (i + 1) % 500 == 0:
            if max_lines_retrieved == -1:
                logging.debug('Has extracted features for %d records (Elapsed: %.1f seconds)',
                              i + 1,
                              (time.time() - start_extracting_time))
            else:
                logging.debug('Has extracted features for %d records (Progress: %.2f %%) (Elapsed: %.1f seconds)',
                              i + 1,
                              ((i+1) * 100 / max_lines_retrieved),
                              (time.time() - start_extracting_time))
        if (i+1) % 1000 == 0 and features_updated:
            checkpoint_features(pickle_path, features)
            features_updated = False

    if features_updated:
        checkpoint_features(pickle_path, features)

    for i, image_id in enumerate(image_id_list):
        if image_id[0] in features:
            result.append(features[image_id[0]][0])
        if max_lines_retrieved != -1 and (i + 1) == max_lines_retrieved:
            break

    return np.array(result)


