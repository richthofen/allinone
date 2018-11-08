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
import datetime
import os
import sql
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
import sys,traceback
def loadCSVfile2(file, line):
    try:
        tmp = numpy.loadtxt(file, dtype=numpy.str )
        # tmp = numpy.loadtxt(file, dtype=numpy.str, delimiter=",", skiprows= 1)
        # print(file)
        # print (tmp.shape)
        times  = tmp[0:line,0].astype(numpy.float)
        global  ori_values 
        ori_values = tmp[:,[2,3,4,5,6,8]].astype(numpy.float)
        values = tmp[0:line,[2,3,4,5,6,8]].astype(numpy.float)
        return {
        feature_keys.TrainEvalFeatures.TIMES: times,
            feature_keys.TrainEvalFeatures.VALUES: values
        }
    except: 
        traceback.print_exception(*sys.exc_info())
        sys.exit()
        return None
def loadNumpyArray(tmp, line):
    try:
        times  = numpy.arange(line)
        # global  ori_values 
        # ori_values = tmp[:,[1,2,3,4,5,6]].astype(numpy.float)
        values = tmp[0:line,[1,2,3,4,5,6]].astype(numpy.float)

        return {
        feature_keys.TrainEvalFeatures.TIMES: times,
            feature_keys.TrainEvalFeatures.VALUES: values
        }
    except: 
        traceback.print_exception(*sys.exc_info())
        sys.exit()
        return None

def multivariate_train_and_sample1(
    data, export_directory=None, training_steps=500, line=600, symbol = None , dt = None):
  # 如果预测过了 不做重复计算
  predict = sql.get_predict(symbol, dt)
  print (predict)
  if 0 != len(predict) :
     return

  """Trains, evaluates, and exports a multivariate model."""
  estimator = tf.contrib.timeseries.StructuralEnsembleRegressor(
      periodicities=[], num_features=6)
  weight = numpy.zeros(11)
  weight [10] = symbol
  data = loadNumpyArray(data, line)
  if None == data:
      print("data get error")
      return
  reader = tf.contrib.timeseries.NumpyReader(data)
  train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
      reader, batch_size=8, window_size=128)
      
  print("start train ")
  estimator.train(input_fn=train_input_fn, steps=training_steps)
  evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
  current_state = estimator.evaluate(input_fn=evaluation_input_fn, steps=1)
  print("after eval ")
  values = [current_state["observed"]]
  times = [current_state[tf.contrib.timeseries.FilteringResults.TIMES]]
  # Export the model so we can do iterative prediction and filtering without
  # reloading model checkpoints.
  if export_directory is None:
    export_directory = tempfile.mkdtemp()
  input_receiver_fn = estimator.build_raw_serving_input_receiver_fn()
  export_location = estimator.export_savedmodel(
      export_directory, input_receiver_fn)
  with tf.Graph().as_default():
    numpy.random.seed(1)  # Make the example a bit more deterministic
    with tf.Session() as session:
      signatures = tf.saved_model.loader.load(
          session, [tf.saved_model.tag_constants.SERVING], export_location)
      global  ori_values 
      predicts = []
      for _ in range(_PRI_NUM):
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
        filtering_features = {
            tf.contrib.timeseries.TrainEvalFeatures.TIMES: current_prediction[
                tf.contrib.timeseries.FilteringResults.TIMES],
            tf.contrib.timeseries.TrainEvalFeatures.VALUES: next_sample[
                None, None, :]}
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
  # pre = pre[:,[0,1,2,3,5]]
  # close = ori_values[line,1] 
#   line += 1
  print ("line of ori %d" %(line))
#   ob = ori_values[line :line + 10,[0,1,2,3,5]].astype(numpy.float)
 
  numpy.set_printoptions(suppress=True)
  numpy.set_numeric_ops
  sql.add_predict(pre, symbol, dt)
  # with open(abs_path, 'ab') as f:
  #   #   tofile = numpy.append(ob, pre, axis=1)
  #     numpy.savetxt(f,pre, fmt='%.2f')
  all_observations = numpy.squeeze(numpy.concatenate(values, axis=1), axis=0)
  all_times = numpy.squeeze(numpy.concatenate(times, axis=1), axis=0)
  return all_times, all_observations
def multivariate_train_and_sample(
    csv_file_name, export_directory=None, training_steps=1, line=600, symbol = None , dt = None):

  abs_path =  path.join(_MODULE_PATH, 'result/' + symbol + "-" + str(line) + "-" + dt +  "pre.csv" )
  if os.path.exists(abs_path):
     return

  """Trains, evaluates, and exports a multivariate model."""
  estimator = tf.contrib.timeseries.StructuralEnsembleRegressor(
      periodicities=[], num_features=6)
  weight = numpy.zeros(11)
  weight [10] = symbol
  data = loadCSVfile2(csv_file_name, line)
  if None == data:
      print("data get error")
      return
  reader = tf.contrib.timeseries.NumpyReader(data)
  train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
      reader, batch_size=8, window_size=128)
      
  print("start train ")
  estimator.train(input_fn=train_input_fn, steps=training_steps)
  evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
  current_state = estimator.evaluate(input_fn=evaluation_input_fn, steps=1)
  print("after eval ")
  values = [current_state["observed"]]
  times = [current_state[tf.contrib.timeseries.FilteringResults.TIMES]]
  # Export the model so we can do iterative prediction and filtering without
  # reloading model checkpoints.
  if export_directory is None:
    export_directory = tempfile.mkdtemp()
  input_receiver_fn = estimator.build_raw_serving_input_receiver_fn()
  export_location = estimator.export_savedmodel(
      export_directory, input_receiver_fn)
  with tf.Graph().as_default():
    numpy.random.seed(1)  # Make the example a bit more deterministic
    with tf.Session() as session:
      signatures = tf.saved_model.loader.load(
          session, [tf.saved_model.tag_constants.SERVING], export_location)
      global  ori_values 
      predicts = []
      for _ in range(_PRI_NUM):
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
        filtering_features = {
            tf.contrib.timeseries.TrainEvalFeatures.TIMES: current_prediction[
                tf.contrib.timeseries.FilteringResults.TIMES],
            tf.contrib.timeseries.TrainEvalFeatures.VALUES: next_sample[
                None, None, :]}
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
  pre = pre[:,[0,1,2,3,5]]
  # close = ori_values[line,1] 
#   line += 1
  print ("line of ori %d" %(line))
#   ob = ori_values[line :line + 10,[0,1,2,3,5]].astype(numpy.float)
 
  numpy.set_printoptions(suppress=True)
  numpy.set_numeric_ops
  # pre = numpy.around(pre, 2)
  # numpy.set_printoptions(suppress=True)
  with open(abs_path, 'ab') as f:
    #   tofile = numpy.append(ob, pre, axis=1)
      numpy.savetxt(f,pre, fmt='%.2f')
  all_observations = numpy.squeeze(numpy.concatenate(values, axis=1), axis=0)
  all_times = numpy.squeeze(numpy.concatenate(times, axis=1), axis=0)
  return all_times, all_observations


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
      pre = sql.get_data(symbol)
      pre = numpy.array(pre)
      l = pre.shape[0] 
      update = False
      for i in range(0, l):
        batch = min(i + 10, l)
        # print (i, batch)
        if float(pre[i,6]) == 0: 
          values = pre[i:batch,[1,2,3,4]].astype(numpy.float)
          x = profit.findTime(values)
          pre[i,6] = pre[x[1] + i ,1]
          # print (x)
          update = True
        # thePoint.append(pre[x[1] + i ,2])
    #   print(thePoint)
    #   print(len(thePoint))
    # 更新best 字段
      if True == update:
        sql.add_tran_data(pre, symbol)
      # print(pre)
      l = pre.shape[0]
      print (pre.shape)
      ##  add column 
      print(symbol)
      for i in range(l - 5, l + 1 , 5):
        print (pre[i - 1,1])
        print (i)
        # continue
        # sys.exit()
        dt = str(pre[i - 1,0])
        multivariate_train_and_sample1(line = i, data = pre , symbol = symbol, dt=dt)
        # multivariate_train_and_sample(line = i, csv_file_name = abs_path, symbol = symbol, dt=dt)

    #   return

if __name__ == "__main__":
  tf.app.run(main=main)
