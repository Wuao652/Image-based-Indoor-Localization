from tensorflow.keras.callbacks import CSVLogger, TensorBoard

import six
import os
import csv
from collections import OrderedDict
from collections import Iterable
import numpy as np
import tensorflow as tf
import sys
# from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

# from keras.callbacks import Callback
# from keras import backend as K

class BatchCSVLogger(CSVLogger):
    """
    Custom callback inheriting from the CSVLogger. This class logs the data at the end of 
    every batch rather than epoch. It manually saves the epoch number and logs this at the
    end of every batch as well.
    """
    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = 'b' if six.PY2 and os.name == 'nt' else ''
        self.epoch = 1
        super(BatchCSVLogger, self).__init__(filename=filename, separator=separator, append=append)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        def handle_value_batch(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(logs.keys())

            class CustomDialect(csv.excel):
                delimiter = self.sep

            print(self.keys)
            print(len(self.keys))
            for i in list(self.keys):
                if(i == 'batch'):
                    self.keys.remove(i)

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch', 'batch'] + self.keys, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': self.epoch})
        row_dict = OrderedDict({'batch': batch})
        row_dict.update((key, handle_value_batch(logs[key])) for key in self.keys)
        row_dict.update((('epoch', handle_value_batch(self.epoch)),))
        self.writer.writerow(row_dict)
        self.csv_file.flush()
        print('\n')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch += 1


# class BatchTensorBoard(TensorBoard):
#     """
#     """

#     def __init__(self, log_dir='./logs',
#                  histogram_freq=0,
#                  batch_size=32,
#                  write_graph=True,
#                  write_grads=False,
#                  write_images=False,
#                  embeddings_freq=0,
#                  embeddings_layer_names=None,
#                  embeddings_metadata=None):
#         super(TensorBoard, self).__init__()
#         self.log_dir = log_dir
#         self.histogram_freq = histogram_freq
#         self.merged = None
#         self.write_graph = write_graph
#         self.write_grads = write_grads
#         self.write_images = write_images
#         self.embeddings_freq = embeddings_freq
#         self.embeddings_layer_names = embeddings_layer_names
#         self.embeddings_metadata = embeddings_metadata or {}
#         self.batch_size = batch_size

#         self.epoch = 0

#         super(BatchTensorBoard, self).__init__(log_dir=log_dir,
#                                                histogram_freq=histogram_freq,
#                                                batch_size=batch_size,
#                                                write_graph=write_graph,
#                                                write_grads=write_grads,
#                                                write_images=write_images,
#                                                embeddings_freq=embeddings_freq,
#                                                embeddings_layer_names=embeddings_layer_names,
#                                                embeddings_metadata=embeddings_metadata)

#     def on_batch_end(self, batch, logs=None):
#         logs = logs or {}

#         if self.validation_data and self.histogram_freq:
#             if batch % self.histogram_freq == 0:

#                 val_data = self.validation_data
#                 tensors = (self.model.inputs +
#                            self.model.targets +
#                            self.model.sample_weights)

#                 if self.model.uses_learning_phase:
#                     tensors += [K.learning_phase()]

#                 assert len(val_data) == len(tensors)
#                 val_size = val_data[0].shape[0]
#                 i = 0
#                 while i < val_size:
#                     step = min(self.batch_size, val_size - i)
#                     if self.model.uses_learning_phase:
#                         # do not slice the learning phase
#                         batch_val = [x[i:i + step] for x in val_data[:-1]]
#                         batch_val.append(val_data[-1])
#                     else:
#                         batch_val = [x[i:i + step] for x in val_data]
#                     assert len(batch_val) == len(tensors)
#                     feed_dict = dict(zip(tensors, batch_val))
#                     result = self.sess.run([self.merged], feed_dict=feed_dict)
#                     summary_str = result[0]
#                     self.writer.add_summary(summary_str, self.epoch * self.batch_size + batch)
#                     i += self.batch_size

#         if self.embeddings_freq and self.embeddings_ckpt_path:
#             if batch % self.embeddings_freq == 0:
#                 self.saver.save(self.sess,
#                                 self.embeddings_ckpt_path,
#                                 batch)

#         for name, value in logs.items():
#             if name in ['batch', 'size']:
#                 continue
#             summary = tf.Summary()
#             summary_value = summary.value.add()
#             summary_value.simple_value = value.item()
#             summary_value.tag = name
#             self.writer.add_summary(summary, self.epoch * self.batch_size + batch)
#         self.writer.flush()

#     def on_epoch_end(self, epoch, logs=None):
#         self.epoch += 1
#         if self.validation_data and self.histogram_freq:
#             if epoch % self.histogram_freq == 0:

#                 val_data = self.validation_data
#                 tensors = (self.model.inputs +
#                            self.model.targets +
#                            self.model.sample_weights)

#                 if self.model.uses_learning_phase:
#                     tensors += [K.learning_phase()]

#                 assert len(val_data) == len(tensors)
#                 val_size = val_data[0].shape[0]
#                 i = 0
#                 while i < val_size:
#                     step = min(self.batch_size, val_size - i)
#                     if self.model.uses_learning_phase:
#                         # do not slice the learning phase
#                         batch_val = [x[i:i + step] for x in val_data[:-1]]
#                         batch_val.append(val_data[-1])
#                     else:
#                         batch_val = [x[i:i + step] for x in val_data]
#                     assert len(batch_val) == len(tensors)
#                     feed_dict = dict(zip(tensors, batch_val))
#                     result = self.sess.run([self.merged], feed_dict=feed_dict)
#                     summary_str = result[0]
#                     self.writer.add_summary(summary_str, epoch)
#                     i += self.batch_size

#         if self.embeddings_freq and self.embeddings_ckpt_path:
#             if epoch % self.embeddings_freq == 0:
#                 self.saver.save(self.sess,
#                                 self.embeddings_ckpt_path,
#                                 epoch)

#         for name, value in logs.items():
#             if name in ['batch', 'size']:
#                 continue
#             summary = tf.Summary()
#             summary_value = summary.value.add()
#             summary_value.simple_value = value.item()
#             summary_value.tag = name
#             self.writer.add_summary(summary, epoch)
#         self.writer.flush()


# # noinspection PySimplifyBooleanCheck
# class TensorBoard(Callback):
#     """Tensorboard basic visualizations.
#     This callback writes a log for TensorBoard, which allows
#     you to visualize dynamic graphs of your training and test
#     metrics, as well as activation histograms for the different
#     layers in your model.
#     TensorBoard is a visualization tool provided with TensorFlow.
#     If you have installed TensorFlow with pip, you should be able
#     to launch TensorBoard from the command line:
#     ```
#     tensorboard --logdir=/full_path_to_your_logs
#     ```
#     You can find more information about TensorBoard
#     [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).
#     # Arguments
#         log_dir: the path of the directory where to save the log
#             files to be parsed by TensorBoard.
#         histogram_freq: frequency (in epochs) at which to compute activation
#             and weight histograms for the layers of the model. If set to 0,
#             histograms won't be computed. Validation data (or split) must be
#             specified for histogram visualizations.
#         write_graph: whether to visualize the graph in TensorBoard.
#             The log file can become quite large when
#             write_graph is set to True.
#         write_grads: whether to visualize gradient histograms in TensorBoard.
#             `histogram_freq` must be greater than 0.
#         batch_size: size of batch of inputs to feed to the network
#             for histograms computation.
#         write_images: whether to write model weights to visualize as
#             image in TensorBoard.
#         write_batch_performance: whether to write training metrics on batch 
#             completion 
#         embeddings_freq: frequency (in epochs) at which selected embedding
#             layers will be saved.
#         embeddings_layer_names: a list of names of layers to keep eye on. If
#             None or empty list all the embedding layer will be watched.
#         embeddings_metadata: a dictionary which maps layer name to a file name
#             in which metadata for this embedding layer is saved. See the
#             [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
#             about metadata files format. In case if the same metadata file is
#             used for all embedding layers, string can be passed.
#     """

#     def __init__(self, log_dir='./logs',
#                  histogram_freq=0,
#                  batch_size=32,
#                  write_graph=True,
#                  write_grads=False,
#                  write_images=False,
#                  write_batch_performance=False,
#                  embeddings_freq=0,
#                  embeddings_layer_names=None,
#                  embeddings_metadata=None):
#         super(TensorBoard, self).__init__()
# #         if K.backend() != 'tensorflow':
# #             raise RuntimeError('TensorBoard callback only works '
# #                                'with the TensorFlow backend.')
#         self.log_dir = log_dir
#         self.histogram_freq = histogram_freq
#         self.merged = None
#         self.write_graph = write_graph
#         self.write_grads = write_grads
#         self.write_images = write_images
#         self.write_batch_performance = write_batch_performance
#         self.embeddings_freq = embeddings_freq
#         self.embeddings_layer_names = embeddings_layer_names
#         self.embeddings_metadata = embeddings_metadata or {}
#         self.batch_size = batch_size
#         self.seen = 0

#     def set_model(self, model):
#         self.model = model
# #         self.sess = K.get_session()
#         if self.histogram_freq and self.merged is None:
#             for layer in self.model.layers:

#                 for weight in layer.weights:
#                     tf.compat.v1.summary.histogram(weight.name, weight)
#                     if self.write_grads:
#                         grads = model.optimizer.get_gradients(model.total_loss,
#                                                               weight)
#                         tf.compat.v1.summary.histogram('{}_grad'.format(weight.name), grads)
#                     if self.write_images:
#                         w_img = tf.squeeze(weight)
#                         shape = K.int_shape(w_img)
#                         if len(shape) == 2:  # dense layer kernel case
#                             if shape[0] > shape[1]:
#                                 w_img = tf.transpose(a=w_img)
#                                 shape = K.int_shape(w_img)
#                             w_img = tf.reshape(w_img, [1,
#                                                        shape[0],
#                                                        shape[1],
#                                                        1])
#                         elif len(shape) == 3:  # convnet case
#                             if K.image_data_format() == 'channels_last':
#                                 # switch to channels_first to display
#                                 # every kernel as a separate image
#                                 w_img = tf.transpose(a=w_img, perm=[2, 0, 1])
#                                 shape = K.int_shape(w_img)
#                             w_img = tf.reshape(w_img, [shape[0],
#                                                        shape[1],
#                                                        shape[2],
#                                                        1])
#                         elif len(shape) == 1:  # bias case
#                             w_img = tf.reshape(w_img, [1,
#                                                        shape[0],
#                                                        1,
#                                                        1])
#                         else:
#                             # not possible to handle 3D convnets etc.
#                             continue

#                         shape = K.int_shape(w_img)
#                         assert len(shape) == 4 and shape[-1] in [1, 3, 4]
#                         tf.compat.v1.summary.image(weight.name, w_img)

#                 if hasattr(layer, 'output'):
#                     tf.compat.v1.summary.histogram('{}_out'.format(layer.name),
#                                          layer.output)
#         self.merged = tf.compat.v1.summary.merge_all()

#         if self.write_graph:
#             self.writer = tf.compat.v1.summary.FileWriter(self.log_dir,
#                                                 self.sess.graph)
#         else:
#             self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)

#         if self.embeddings_freq:
#             embeddings_layer_names = self.embeddings_layer_names

#             if not embeddings_layer_names:
#                 embeddings_layer_names = [layer.name for layer in self.model.layers
#                                           if type(layer).__name__ == 'Embedding']

#             embeddings = {layer.name: layer.weights[0]
#                           for layer in self.model.layers
#                           if layer.name in embeddings_layer_names}

#             self.saver = tf.compat.v1.train.Saver(list(embeddings.values()))

#             embeddings_metadata = {}

#             if not isinstance(self.embeddings_metadata, str):
#                 embeddings_metadata = self.embeddings_metadata
#             else:
#                 embeddings_metadata = {layer_name: self.embeddings_metadata
#                                        for layer_name in embeddings.keys()}

#             config = projector.ProjectorConfig()
#             self.embeddings_ckpt_path = os.path.join(self.log_dir,
#                                                      'keras_embedding.ckpt')

#             for layer_name, tensor in embeddings.items():
#                 embedding = config.embeddings.add()
#                 embedding.tensor_name = tensor.name

#                 if layer_name in embeddings_metadata:
#                     embedding.metadata_path = embeddings_metadata[layer_name]

#             projector.visualize_embeddings(self.writer, config)

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         if self.validation_data and self.histogram_freq:
#             if epoch % self.histogram_freq == 0:

#                 val_data = self.validation_data
#                 tensors = (self.model.inputs +
#                            self.model.targets +
#                            self.model.sample_weights)

#                 if self.model.uses_learning_phase:
#                     tensors += [K.learning_phase()]

#                 assert len(val_data) == len(tensors)
#                 val_size = val_data[0].shape[0]
#                 i = 0
#                 while i < val_size:
#                     step = min(self.batch_size, val_size - i)
#                     batch_val = []
#                     batch_val.append(val_data[0][i:i + step])
#                     batch_val.append(val_data[1][i:i + step])
#                     batch_val.append(val_data[2][i:i + step])
#                     if self.model.uses_learning_phase:
#                         batch_val.append(val_data[3])
#                     feed_dict = dict(zip(tensors, batch_val))
#                     result = self.sess.run([self.merged], feed_dict=feed_dict)
#                     summary_str = result[0]
#                     self.writer.add_summary(summary_str, self.seen)
#                     i += self.batch_size

#         if self.embeddings_freq and self.embeddings_ckpt_path:
#             if epoch % self.embeddings_freq == 0:
#                 self.saver.save(self.sess,
#                                 self.embeddings_ckpt_path,
#                                 epoch)

#         for name, value in logs.items():
#             if name in ['batch', 'size']:
#                 continue
#             summary = tf.compat.v1.Summary()
#             summary_value = summary.value.add()
#             summary_value.simple_value = value.item()
#             summary_value.tag = name
#             self.writer.add_summary(summary, self.seen)
#         self.writer.flush()
#         self.seen += self.batch_size

#     def on_train_end(self, _):
#         self.writer.close()

#     def on_batch_end(self, batch, logs=None):
#         logs = logs or {}

#         if self.write_batch_performance == True:
#             for name, value in logs.items():
#                 if name in ['batch','size']:
#                     continue
#                 summary = tf.compat.v1.Summary()
#                 summary_value = summary.value.add()
#                 summary_value.simple_value = value.item()
#                 summary_value.tag = name
#                 self.writer.add_summary(summary, self.seen)
#             self.writer.flush()

#         self.seen += self.batch_size

class TensorBoard(Callback):
    """Tensorboard basic visualizations.
    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.
    TensorBoard is a visualization tool provided with TensorFlow.
    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```
    tensorboard --logdir=/full_path_to_your_logs
    ```
    You can find more information about TensorBoard
    [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).
    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation
            and weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard.
            `histogram_freq` must be greater than 0.
        batch_size: size of batch of inputs to feed to the network
            for histograms computation.
        write_images: whether to write model weights to visualize as
            image in TensorBoard.
        write_batch_performance: whether to write training metrics on batch 
            completion 
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
    """

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 write_batch_performance=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None):
        super(TensorBoard, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.write_images = write_images
        self.write_batch_performance = write_batch_performance
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata or {}
        self.batch_size = batch_size
        self.seen = 0

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    tf.summary.histogram(weight.name, weight)
                    if self.write_grads:
                        grads = model.optimizer.get_gradients(model.total_loss,
                                                              weight)
                        tf.summary.histogram('{}_grad'.format(weight.name), grads)
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = K.int_shape(w_img)
                        if len(shape) == 2:  # dense layer kernel case
                            if shape[0] > shape[1]:
                                w_img = tf.transpose(w_img)
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       shape[1],
                                                       1])
                        elif len(shape) == 3:  # convnet case
                            if K.image_data_format() == 'channels_last':
                                # switch to channels_first to display
                                # every kernel as a separate image
                                w_img = tf.transpose(w_img, perm=[2, 0, 1])
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0],
                                                       shape[1],
                                                       shape[2],
                                                       1])
                        elif len(shape) == 1:  # bias case
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       1,
                                                       1])
                        else:
                            # not possible to handle 3D convnets etc.
                            continue

                        shape = K.int_shape(w_img)
                        assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                        tf.summary.image(weight.name, w_img)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name),
                                         layer.output)
        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.embeddings_freq:
            embeddings_layer_names = self.embeddings_layer_names

            if not embeddings_layer_names:
                embeddings_layer_names = [layer.name for layer in self.model.layers
                                          if type(layer).__name__ == 'Embedding']

            embeddings = {layer.name: layer.weights[0]
                          for layer in self.model.layers
                          if layer.name in embeddings_layer_names}

            self.saver = tf.train.Saver(list(embeddings.values()))

            embeddings_metadata = {}

            if not isinstance(self.embeddings_metadata, str):
                embeddings_metadata = self.embeddings_metadata
            else:
                embeddings_metadata = {layer_name: self.embeddings_metadata
                                       for layer_name in embeddings.keys()}

            config = projector.ProjectorConfig()
            self.embeddings_ckpt_path = os.path.join(self.log_dir,
                                                     'keras_embedding.ckpt')

            for layer_name, tensor in embeddings.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name

                if layer_name in embeddings_metadata:
                    embedding.metadata_path = embeddings_metadata[layer_name]

            projector.visualize_embeddings(self.writer, config)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:

                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    batch_val = []
                    batch_val.append(val_data[0][i:i + step])
                    batch_val.append(val_data[1][i:i + step])
                    batch_val.append(val_data[2][i:i + step])
                    if self.model.uses_learning_phase:
                        batch_val.append(val_data[3])
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, self.seen)
                    i += self.batch_size

        if self.embeddings_freq and self.embeddings_ckpt_path:
            if epoch % self.embeddings_freq == 0:
                self.saver.save(self.sess,
                                self.embeddings_ckpt_path,
                                epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.seen)
        self.writer.flush()
        self.seen += self.batch_size

    def on_train_end(self, _):
        self.writer.close()

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        if self.write_batch_performance == True:
            for name, value in logs.items():
                if name in ['batch','size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.seen)
            self.writer.flush()

        self.seen += self.batch_size