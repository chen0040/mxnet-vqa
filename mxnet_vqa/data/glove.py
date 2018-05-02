from mxnet_vqa.utils.glove_loader import load_glove


def glove_word2emb_300(data_dir_path):
    load_glove(data_dir_path, embedding_dim=300)