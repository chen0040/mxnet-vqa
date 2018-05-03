import mxnet as mx
from mxnet import gluon, autograd, nd
import mxnet.ndarray.contrib as C
from mxnet.gluon import nn
import os
import numpy as np
import logging


class Net2(gluon.Block):

    def __init__(self, model_ctx, out_dim, nb_classes, batch_size, **kwargs):
        super(Net2, self).__init__(**kwargs)
        self.model_ctx = model_ctx
        self.out_dim = out_dim
        self.batch_size = batch_size
        with self.name_scope():
            self.bn = nn.BatchNorm()
            self.dropout = nn.Dropout(.3)
            self.fc1 = nn.Dense(8192, activation='relu')
            self.fc2 = nn.Dense(nb_classes)

    def forward(self, x, *args, **kwargs):
        F = nd

        x1 = F.L2Normalization(x[0])
        x2 = F.L2Normalization(x[1])

        x1 = F.concat(x1, F.ones((self.batch_size, 128), ctx=self.model_ctx), dim=1)
        x2 = F.concat(x2, F.ones((self.batch_size, 128), ctx=self.model_ctx), dim=1)

        x1_dim = x1.shape[1]
        x2_dim = x2.shape[1]

        # Initialize hash tables
        S1 = F.array(np.random.randint(0, 2, (1, x1_dim))*2-1, ctx=self.model_ctx)
        H1 = F.array(np.random.randint(0, self.out_dim, (1, x1_dim)), ctx=self.model_ctx)
        S2 = F.array(np.random.randint(0, 2, (1, x2_dim))*2-1, ctx=self.model_ctx)
        H2 = F.array(np.random.randint(0, self.out_dim, (1, x2_dim)), ctx=self.model_ctx)
        # Count sketch
        cs1 = C.count_sketch(data=x1, s=S1, h=H1, name='cs1', out_dim=self.out_dim)
        cs2 = C.count_sketch(data=x2, s=S2, h=H2, name='cs2', out_dim=self.out_dim)

        fft1 = C.fft(data=cs1, name='fft1', compute_size=self.batch_size)
        fft2 = C.fft(data=cs2, name='fft2', compute_size=self.batch_size)
        c = fft1 * fft2
        ifft1 = C.ifft(data=c, name='ifft1', compute_size=self.batch_size)
        # MLP
        z = self.fc1(ifft1)
        z = self.bn(z)
        z = self.dropout(z)
        z = self.fc2(z)
        return z


class VQANet(object):
    model_name = 'vqa-net-2'

    def __init__(self, model_ctx=mx.cpu(), data_ctx=mx.cpu()):
        self.model = None
        self.version = '0'
        self.model_ctx = model_ctx
        self.data_ctx = data_ctx
        self.input_mode_answer = 'int'
        self.input_mode_question = 'add'
        self.nb_classes = 1001
        self.batch_size = 64
        self.out_dim = 10000
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
        self.batch_size = config['batch_size']
        self.meta = config['meta']
        self.out_dim = config['out_dim']
        self.model = Net2(model_ctx=self.model_ctx, out_dim=self.out_dim, nb_classes=self.nb_classes, batch_size=self.batch_size)
        self.model.load_params(self.get_params_file_path(model_dir_path), ctx=self.model_ctx)

    def checkpoint(self, model_dir_path):
        self.model.save_params(self.get_params_file_path(model_dir_path))

    def save_history(self, history, model_dir_path):
        return np.save(os.path.join(model_dir_path, VQANet.model_name + '-v' + self.version + '-history.npy'), history)

    def fit(self, data_train, data_eva, meta, model_dir_path, epochs=10, batch_size=64, learning_rate=0.01):

        self.batch_size = batch_size

        config = dict()
        config['batch_size'] = batch_size
        config['input_mode_answer'] = self.input_mode_answer
        config['input_mode_question'] = self.input_mode_question
        config['nb_classes'] = self.nb_classes
        config['out_dim'] = self.out_dim
        config['meta'] = meta
        np.save(self.get_config_file_path(model_dir_path), config)

        loss = gluon.loss.SoftmaxCrossEntropyLoss()

        self.model = Net2(model_ctx=self.model_ctx, out_dim=self.out_dim, batch_size=batch_size, nb_classes=self.nb_classes)
        self.model.collect_params().initialize(init=mx.init.Xavier(), ctx=self.model_ctx)
        trainer = gluon.Trainer(self.model.collect_params(), 'adam', {'learning_rate': learning_rate})

        history = dict()
        history['train_acc'] = list()
        history['val_acc'] = list()

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
                trainer.step(batch_size)

                if i == 0:
                    moving_loss = np.mean(cross_entropy.asnumpy()[0])
                else:
                    moving_loss = .99 * moving_loss + .01 * np.mean(cross_entropy.asnumpy()[0])
                if i % 200 == 0:
                    logging.debug("Epoch %s, batch %s. Moving avg of loss: %s", e, i, moving_loss)
            eva_accuracy = self.evaluate_accuracy(data_iterator=data_eva)
            train_accuracy = self.evaluate_accuracy(data_iterator=data_train)
            history['train_acc'].append(train_accuracy)
            history['val_acc'].append(eva_accuracy)
            print("Epoch %s. Loss: %s, Train_acc %s, Eval_acc %s" % (e, moving_loss, train_accuracy, eva_accuracy))
            if eva_accuracy > best_eva:
                best_eva = eva_accuracy
                logging.info('Best validation acc found. Checkpointing...')
                self.checkpoint(model_dir_path)

        self.save_history(history, model_dir_path)
        return history
