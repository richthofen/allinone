# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A multivariate TFTS example.
Fits a multivariate model, exports it, and visualizes the learned correlations
by iteratively predicting and sampling from the predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib.timeseries.python.timeseries import feature_keys
from os import path
import tempfile
import numpy
import tensorflow as tf
import evaluate
import profit
import os
import datetime
import pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
try:
  import matplotlib  # pylint: disable=g-import-not-at-top
  matplotlib.use("TkAgg")  # Need Tk for interactive plots.
  from matplotlib import pyplot  # pylint: disable=g-import-not-at-top
  HAS_MATPLOTLIB = True
except ImportError:
  # Plotting requires matplotlib, but the unit test running this code may
  # execute in an environment without it (i.e. matplotlib is not a build
  # dependency). We'd still like to test the TensorFlow-dependent parts of this
  # example, namely train_and_predict.
  HAS_MATPLOTLIB = False

_MODULE_PATH = path.dirname(__file__)
# _DATA_FILE = path.join(_MODULE_PATH, "600887.csv")
_PRI_NUM = 10
# _DATA_FILE = path.join(_MODULE_PATH, "cc.log")
ori_values = []

tofile = ""
def loadCSVfile2(file, line):
    try:
        tmp = numpy.loadtxt(file, dtype=numpy.str, delimiter=",", skiprows= 1)
        times  = tmp[0:line,0].astype(numpy.float)
        global  ori_values 
        ori_values = tmp[:,[2,3,4,5,6]].astype(numpy.float)
        values = tmp[0:line,[2,3,4,5,6]].astype(numpy.float)
        # filter stop  part 
        last_date = tmp[-1, 1]
        last_day = datetime.datetime.strptime(last_date, '%Y-%m-%d')
        now = datetime.datetime.now()
        diff_day = 1
        if now.weekday() == 0:
            diff_day = 3
        delta = now - last_day
        if diff_day < delta.days :
            return None
        return {
        feature_keys.TrainEvalFeatures.TIMES: times,
            feature_keys.TrainEvalFeatures.VALUES: values
        }
    except:
        return None

def multivariate_train_and_sample(
    csv_file_name, export_directory=None, training_steps=5, line=600, symbol = None):
  """Trains, evaluates, and exports a multivariate model."""
#   estimator = tf.contrib.timeseries.StructuralEnsembleRegressor(
#       periodicities=[], num_features=5)
#   weight = numpy.zeros(11)
#   weight [10] = symbol
#   data = loadCSVfile2(csv_file_name, line)
#   if None == data:
#       print("data get error")
#       return
#   reader = tf.contrib.timeseries.NumpyReader(data)
#   train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
#       # Larger window sizes generally produce a better covariance matrix.
#       reader, batch_size=8, window_size=128)
      
#   print("start train ")
#   estimator.train(input_fn=train_input_fn, steps=training_steps)
#   evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
#   current_state = estimator.evaluate(input_fn=evaluation_input_fn, steps=1)
#   print("after eval ")
#   values = [current_state["observed"]]
#   times = [current_state[tf.contrib.timeseries.FilteringResults.TIMES]]
#   # Export the model so we can do iterative prediction and filtering without
#   # reloading model checkpoints.
#   if export_directory is None:
#     export_directory = tempfile.mkdtemp()
#   input_receiver_fn = estimator.build_raw_serving_input_receiver_fn()
#   export_location = estimator.export_savedmodel(
#       export_directory, input_receiver_fn)
  export_location = 'tmp_model/' + symbol
  with tf.Graph().as_default():
    numpy.random.seed(1)  # Make the example a bit more deterministic
    with tf.Session() as session:
      signatures = tf.saved_model.loader.load(
          session, [tf.saved_model.tag_constants.SERVING], export_location)
      global  ori_values 
      predicts = []
      filtering_features = {
        tf.contrib.timeseries.TrainEvalFeatures.TIMES: current_prediction[
            tf.contrib.timeseries.FilteringResults.TIMES],
        tf.contrib.timeseries.TrainEvalFeatures.VALUES: next_sample[
            None, None, :]}
      for _ in range(_PRI_NUM):
        state = tf.contrib.timeseries.saved_model_utils.cold_start_filter(
          signatures=signatures, session=session, features=filtering_features)
        current_prediction = (
            tf.contrib.timeseries.saved_model_utils.predict_continuation(
                continue_from=current_state, signatures=signatures,
                session=session, steps=1))
        next_sample = numpy.random.multivariate_normal(
            # Squeeze out the batch and series length dimensions (both 1).
            mean=numpy.squeeze(current_prediction["mean"], axis=[0, 1]),
            cov=numpy.squeeze(current_prediction["covariance"], axis=[0, 1]))
        # Update model state so that future predictions are conditional on the
        # value we just sampled.
        current_state = (
            tf.contrib.timeseries.saved_model_utils.filter_continuation(
                continue_from=current_state,
                session=session,
                signatures=signatures,
                features=filtering_features))
        values.append(next_sample[None, None, :])
        predicts.append(next_sample[None, None, :])
        times.append(current_state["times"])

  pre = numpy.array(predicts)
  pre = numpy.squeeze(pre)
  pre = pre[:,[0,1,2,3]]
  print (pre)
  ob = ori_values[line :,[0,1,2,3]].astype(numpy.float)
  with open('tmp_data/' + symbol + 'ob' + '_stage' + str(line) + '.csv', 'wb') as f:
    numpy.savetxt(f, ob)
  with open('tmp_data/' + symbol + 'pre' + '_stage' + str(line) + '.csv', 'wb') as f:
    numpy.savetxt(f, pre)
  return pre, ob


def calPredict(symbol, predictos, close):
  numpy.set_printoptions(suppress=True)
  preProfit,buy,sell = profit.findTime(predictos)
  global tofile
  preBuy = predictos[buy, 3]
  start = min(preBuy, close)
  if preProfit > 0.06 and buy < 1.01:
      tofile += "%s, profit:%f, in:%f, buy day@%f, sell day@%f\n" %(symbol, preProfit, start, buy, sell)
def readObPre(symbol, line):
    stage_ob_file = pathlib.Path('tmp_data/' + symbol + 'ob' + '_stage' + str(line) + '.csv')
    stage_pre_file = pathlib.Path('tmp_data/' + symbol + 'pre' + '_stage' + str(line) + '.csv')
    ob = None
    pre = None
    if stage_ob_file.is_file():
        ob = numpy.loadtxt(stage_ob_file) 
    if stage_pre_file.is_file():
        pre = numpy.loadtxt(stage_pre_file) 
    return ob, pre
def main(unused_argv):
  if not HAS_MATPLOTLIB:
    raise ImportError(
        "Please install matplotlib to generate a plot from this example.")
  abs_path =  path.join(_MODULE_PATH, 'back')
  with open(abs_path) as file:
    data = file.read().split("\n",113)
    for symbol in data:
      if '' == symbol:
        break
      print(symbol)
      l = 0
      abs_path =  path.join(_MODULE_PATH, 'data/' + symbol + ".csv" )
      try:
        with open( abs_path) as f:
            l = len(f.readlines())
            print(symbol)
            line = l - 6
            stage_ob_file = pathlib.Path('tmp_data/' + symbol + 'ob' + '_stage' + str(line) + '.csv')
            stage_pre_file = pathlib.Path('tmp_data/' + symbol + 'pre' + '_stage' + str(line) + '.csv')
            print(stage_ob_file)
            ob, pre = readObPre(symbol, line)
            print(pre)
            if pre is None:
                pre,ob = multivariate_train_and_sample(line = line, csv_file_name = abs_path, symbol = symbol)
            obj = evaluate.mse(ob, pre)
            print("obj ")
            #   if obj.sum() < 2.5:
            if True:
                line = l - 1
                _, pre = readObPre(symbol, line)
                print (pre)
                if pre is None:
                    pre, _ = multivariate_train_and_sample(line = line, csv_file_name = abs_path, symbol = symbol)
                close = ob[-1, 1]
                calPredict(symbol, pre, close)
      except Exception as e:
          print(e)
  predict_path =  path.join(_MODULE_PATH, 'predict' )
  with open(predict_path, 'w') as f:
      global tofile

      f.write(tofile)

if __name__ == "__main__":
  tf.app.run(main=main)
