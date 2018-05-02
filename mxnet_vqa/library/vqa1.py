import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
import os
import numpy as np
import logging


class Net1(gluon.Block):

    def __init__(self, nb_classes, **kwargs):
        super(Net1, self).__init__(**kwargs)
        self.nb_classes = nb_classes
        with self.name_scope():
            self.bn = nn.BatchNorm()
            self.dropout = nn.Dropout(.3)
            self.fc1 = nn.Dense(8192, activation='relu')
            self.fc2 = nn.Dense(self.nb_classes)

    def forward(self, x, *args, **kwargs):
        F = nd
        x1 = F.L2Normalization(x[0])
        x2 = F.L2Normalization(x[1])
        z = F.concat(x1, x2, dim=1)
        z = self.fc1(z)
        z = self.bn(z)
        z = self.dropout(z)
        z = self.fc2(z)
        return z


class VQANet(object):
    model_name = 'vqa-net-1'

    def __init__(self, model_ctx=mx.cpu(), data_ctx=mx.cpu()):
        self.model = None
        self.version = '0'
        self.model_ctx = model_ctx
        self.data_ctx = data_ctx
        self.input_mode_answer = 'int'
        self.input_mode_question = 'add'
        self.nb_classes = 1001
        self.meta = None

    def get_config_file_path(self, model_dir_path):
        return os.path.join(model_dir_path, VQANet.model_name + '-v' + self.version + '-config.npy')

    def get_params_file_path(self, model_dir_path):
        return os.path.join(model_dir_path, VQANet.model_name + '-v' + self.version + '-net.params')

    def evaluate_accuracy(self, data_iterator):
        metric = mx.metric.Accuracy()
        data_iterator.reset()
        for i, batch in enumerate(data_iterator):
            data1 = batch.data[0].as_in_context(self.model_ctx)
            data2 = batch.data[1].as_in_context(self.model_ctx)
            data = [data1, data2]
            label = batch.label[0].as_in_context(self.model_ctx)
            output = self.model(data)

            # metric.update(preds=output, labels=label)
            metric.update([label], [output])
        return metric.get()[1]

    def load_model(self, model_dir_path):
        config = np.load(self.get_config_file_path(model_dir_path)).item()
        self.input_mode_answer = config['input_mode_answer']
        self.input_mode_question = config['input_mode_question']
        self.nb_classes = config['nb_classes']
        self.meta = config['meta']
        self.model = Net1(self.nb_classes)
        self.model.load_params(self.get_params_file_path(model_dir_path), ctx=self.model_ctx)

    def checkpoint(self, model_dir_path):
        self.model.save_params(self.get_params_file_path(model_dir_path))

    def fit(self, data_train, data_eva, meta, model_dir_path, epochs=10, learning_rate=0.01):

        config = dict()
        config['input_mode_answer'] = self.input_mode_answer
        config['input_mode_question'] = self.input_mode_question
        config['nb_classes'] = self.nb_classes
        config['meta'] = meta
        self.meta = meta
        np.save(self.get_config_file_path(model_dir_path), config)

        loss = gluon.loss.SoftmaxCrossEntropyLoss()

        self.model = Net1(self.nb_classes)
        self.model.collect_params().initialize(init=mx.init.Xavier(), ctx=self.model_ctx)
        trainer = gluon.Trainer(self.model.collect_params(), 'sgd', {'learning_rate': learning_rate})

        moving_loss = 0.
        best_eva = 0
        for e in range(epochs):
            data_train.reset()
            for i, batch in enumerate(data_train):
                batch_size = batch.data[0].shape[0]

                data1 = batch.data[0].as_in_context(self.model_ctx)
                data2 = batch.data[1].reshape((batch_size, -1)).as_in_context(self.model_ctx)
                data = [data1, data2]
                label = batch.label[0].as_in_context(self.model_ctx)
                with autograd.record():
                    output = self.model(data)
                    cross_entropy = loss(output, label)
                    cross_entropy.backward()
                trainer.step(batch_size)

                if i == 0:
                    moving_loss = np.mean(cross_entropy.asnumpy()[0])
                else:
                    moving_loss = .99 * moving_loss + .01 * np.mean(cross_entropy.asnumpy()[0])
                if i % 200 == 0:
                    logging.debug("Epoch %s, batch %s. Moving avg of loss: %s", e, i, moving_loss)
            eva_accuracy = self.evaluate_accuracy(data_iterator=data_eva)
            train_accuracy = self.evaluate_accuracy(data_iterator=data_train)
            print("Epoch %s. Loss: %s, Train_acc %s, Eval_acc %s" % (e, moving_loss, train_accuracy, eva_accuracy))
            if eva_accuracy > best_eva:
                best_eva = eva_accuracy
                logging.info('Best validation acc found. Checkpointing...')
                self.checkpoint(model_dir_path)

