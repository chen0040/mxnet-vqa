import mxnet as mx
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon.utils import download
from mxnet import image
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
import os
import numpy as np
import logging


class Net1(gluon.HybridBlock):

    def __init__(self, **kwargs):
        super(Net1, self).__init__(**kwargs)
        with self.name_scope():
            self.bn = nn.BatchNorm()
            self.dropout = nn.Dropout(.3)
            self.fc1 = nn.Dense(8192, activation='relu')
            self.fc2 = nn.Dense(1000)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x1 = F.L2Normalization(x[0])
        x2 = F.L2Normalization(x[1])
        z = F.concat(x1, x2, dim=1)
        z = self.fc1(z)
        z = self.bn(z)
        z = self.dropout(z)
        z = self.fc2(z)
        return z


class VQANet():
    model_name = 'vqa-net-1'

    def __init__(self, model_ctx=mx.cpu(), data_ctx=mx.cpu()):
        self.model = None
        self.image_net = models.vgg16(pretrained=True)
        self.model_ctx = model_ctx
        self.data_ctx = data_ctx

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, VQANet.model_name + '-config.npy')

    @staticmethod
    def get_params_file_path(model_dir_path):
        return os.path.join(model_dir_path, VQANet.model_name + '-net.params')

    def evaluate_accuracy(self, metric, data_iterator):
        data_iterator.reset()
        for i, batch in enumerate(data_iterator):
            with autograd.record():
                data1 = batch.data[0].as_in_context(self.model_ctx)
                data2 = batch.data[1].as_in_context(self.model_ctx)
                data = [data1, data2]
                label = batch.label[0].as_in_context(self.model_ctx)
                output = self.model(data)

            metric.update([label], [output])
        return metric.get()[1]

    def load_model(self, model_dir_path):
        config = np.load(self.get_config_file_path(model_dir_path)).item()
        self.model = Net1()
        self.model.load_params(self.get_params_file_path(model_dir_path), ctx=self.model_ctx)

    def checkpoint(self, model_dir_path):
        self.model.save_params(self.get_params_file_path(model_dir_path))

    def fit(self, data_train, data_eva, model_dir_path, epochs=10, batch_size=32, learning_rate=0.01):

        config = dict()
        np.save(self.get_config_file_path(model_dir_path), config)

        loss = gluon.loss.SoftmaxCrossEntropyLoss()

        metric = mx.metric.Accuracy()

        self.model = Net1()
        self.model.collect_params().initialize(init=mx.init.Xavier(2.24), ctx=self.model_ctx)
        trainer = gluon.Trainer(self.model.collect_params(), 'sgd', {'learning_rate': learning_rate})

        moving_loss = 0.
        best_eva = 0
        for e in range(epochs):
            data_train.reset()
            for i, batch in enumerate(data_train):
                data1 = batch.data[0].as_in_context(self.model_ctx)
                data2 = batch.data[1].as_in_context(self.model_ctx)
                data = [data1, data2]
                label = batch.label[0].as_in_context(self.model_ctx)
                with autograd.record():
                    output = self.model(data)
                    cross_entropy = loss(output, label)
                    cross_entropy.backward()
                trainer.step(data[0].shape[0])

                if i == 0:
                    moving_loss = np.mean(cross_entropy.asnumpy()[0])
                else:
                    moving_loss = .99 * moving_loss + .01 * np.mean(cross_entropy.asnumpy()[0])
                # if i % 200 == 0:
                #    print("Epoch %s, batch %s. Moving avg of loss: %s" % (e, i, moving_loss))
            eva_accuracy = self.evaluate_accuracy(metric=metric, data_iterator=data_eva)
            train_accuracy = self.evaluate_accuracy(metric=metric, data_iterator=data_train)
            print("Epoch %s. Loss: %s, Train_acc %s, Eval_acc %s" % (e, moving_loss, train_accuracy, eva_accuracy))
            if eva_accuracy > best_eva:
                best_eva = eva_accuracy
                logging.info('Best validation acc found. Checkpointing...')
                self.checkpoint(model_dir_path)
