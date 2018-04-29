import os
import pandas as pd
import sys

def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def main():
    sys.path.append(patch_path('..'))
    data_dir_path = patch_path('data/coco')
    data_path = os.path.join(data_dir_path, 'data/val_qa')

    df = pd.read_pickle(data_path)
    print(df.head())

    from mxnet_vqa.data.coco import get_questions, get_coco_features
    questions = get_questions(data_dir_path)
    print('questions: ', len(questions))

    features = get_coco_features(data_dir_path)
    print('features: ', features.shape)



if __name__ == '__main__':
    main()
