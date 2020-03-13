#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# custom callbacks
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#******************************************************************************

import os
import keras
from keras.callbacks import Callback, TensorBoard
import tensorflow as tf
from util.plot import Plotter, LossPlotter
import numpy as np

#---------------------------------------------------------------------------------
class PlotVectorsCallback(Callback):
    def __init__(self, model, x):
        self._model = model
        self._x = x[np.newaxis, ...]
        self._plotter = Plotter()

    def on_epoch_end(self, acc, loss):
        self._y = self._model.predict(self._x)
        self._plotter.plot_vector_field(self._x[0], self._y[0])
        self._plotter.show(False)

#---------------------------------------------------------------------------------
class PlotRealsCallback(Callback):
    def __init__(self, model, x, output_directory, filename):
        self._model = model
        self._x = x[np.newaxis, ...]
        self._filename = filename
        self._out_dir = output_directory
        self._plotter = Plotter()

    def on_epoch_end(self, acc, loss):
        self._y = self._model.predict(self._x)
        self._plotter.plot_heatmap(self._x[0], self._y[0])
        self._plotter.show(False)
        self._plotter.save_figures(self._out_dir, filename=self._filename, filetype="png")

#---------------------------------------------------------------------------------
class PlotRealsAndSlicesCallback(Callback):
    def __init__(self, model, x, output_directory, filenames):
        self._model = model
        self._x = x[np.newaxis, ...]
        self._filenames = filenames
        assert len(filenames) == 2, 'Filenames must have two names'
        self._out_dir = output_directory
        self._plotter_real = Plotter()
        self._plotter_slice = Plotter()

    def on_epoch_end(self, acc, loss):
        self._y = self._model.predict(self._x)
        self._plotter_real.plot_heatmap(self._x[0], self._y[0])
        self._plotter_real.show(False)
        self._plotter_real.save_figures(self._out_dir, filename=self._filenames[0], filetype="png")
        self._plotter_slice.plot_slice(self._x[0], self._y[0], x_slice=0.5)
        self._plotter_slice.show(False)
        self._plotter_slice.save_figures(self._out_dir, filename=self._filenames[1], filetype="png")

#---------------------------------------------------------------------------------
class StatefulResetCallback(Callback):
    def __init__(self, model):
        self.model = model
        self.counter = 0
        
    def on_batch_end(self, batch, logs={}):
        self.counter = self.counter + 1
        if self.counter % 2 == 0:
            print("Resetting states")
            self.model.reset_states()

#---------------------------------------------------------------------------------
class LossHistory(Callback):
    def __init__(self, plot_callback):
        assert plot_callback is not None, "plot_callback can not be 'None'"
        self.plot_callback = plot_callback

    def on_train_begin(self, logs={}):
        self.train_losses = {}
        self.val_losses = {}

    def on_epoch_end(self, epoch, logs):
        self.train_losses[epoch] = logs.get('loss')
        self.val_losses[epoch] = logs.get('val_loss')
        self.plot_callback(self.train_losses, self.val_losses)

class PlotLosses(Callback):
    def __init__(self, path, filename):
        self.plot_callback = LossPlotter(path, filename)

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        self.plot_callback(self.x, self.losses, self.val_losses) 

# from https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure
class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
