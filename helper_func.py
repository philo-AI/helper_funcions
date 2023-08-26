import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np
import sklearn 
def image_imports():
        a = 'import numpy as np'
        b='import pandas as pd'
        c='import matplotlib.pyplot as plt'
        d='import matplotlib.image as img'
        e='import cv2 as cv'
        f='import tensorflow as tf'
        j='from tensorflow import keras'
        h = 'from tensorflow.keras import layers'
        print(a)
        print(b)
        print(c)
        print(d)
        print(e)
        print(f)
        print(j)
        print(h)

def nlp_imports():
        a = 'import numpy as np'
        b='import pandas as pd'
        c='import matplotlib.pyplot as plt'
        d='import matplotlib.image as img'
        f='import tensorflow as tf'
        j='from tensorflow import keras'
        h ='from tensorflow.keras import layers'
        e = 'import tensorflow_hub'
        print(a)
        print(b)
        print(c)
        print(d)
        print(e)
        print(f)
        print(j)
        print(h)

def SplitNlp(x,y,train_s):
        from sklearn.model_selection import train_test_split
        x_train, y_train, x_test, y_test= train_test_split(x,y ,train_size=train_s,shuffle=True)
        print(f"train are{x_train}&{y_train} test are {x_test}&{y_test} ")
        return x_train,y_train ,x_test,y_test

def SplitImage(data,train_s):
      from sklearn.model_selection import train_test_split
      train, test = train_test_split(data,train_size=train_s)
      print(f"train is{train} test is {test} ")
      return(train,test)

def unzip(file_to_unzip , place_to_unzip):
    import zipfile
    with zipfile.ZipFile(file_to_unzip, 'r') as zip_ref:
        zip_ref.extractall(place_to_unzip)

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  import itertools
  import matplotlib.pyplot as plt
  import numpy as np
  from sklearn.metrics import confusion_matrix

  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
  n_classes = cm.shape[0]

  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues)
  fig.colorbar(cax)

  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes),
         yticks=np.arange(n_classes), 
         xticklabels=labels,
         yticklabels=labels)
  
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  threshold = (cm.max() + cm.min()) / 2.

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  if savefig:
    fig.savefig("confusion_matrix.png")

def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

def plot_loss_curves(history):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['Accuracy']
  val_accuracy = history.history['val_Accuracy']

  epochs = range(len(history.history['loss']))

  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  plt.figure()
  plt.plot(epochs, accuracy, label='training_Accuracy')
  plt.plot(epochs, val_accuracy, label='val_Accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend()

def compare_historys(original_history, new_history, initial_epochs=5):    
    acc = original_history.history["Accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_Accuracy"]
    val_loss = original_history.history["val_loss"]

    total_acc = acc + new_history.history["Accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_Accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

def walk_through_dir(dir_path):
  import os
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def calculate_results(y_true, y_pred):
  from sklearn.metrics import accuracy_score, precision_recall_fscore_support
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results

def time_preprocces(dataframe,cost_name,tts):
    timesteps = dataframe.index.to_numpy()
    prices = dataframe[str(cost_name)].to_numpy()
    def SplitTimeSeries(price=prices, timestep=timesteps,tts=tts):
      split_size = int(tts * len(price))
      x_train, y_train = timestep[:split_size], price[:split_size]
      x_test, y_test = timestep[split_size:], price[split_size:]
      print (f"the vals are x_train = {x_train},x_test = {x_test},y_train = {y_train},y_test = {y_test}")
      return x_train, y_train,x_test,y_test

def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("BTC Price")
    if label:
      plt.legend(fontsize=14)
    plt.grid(True)

def mean_absolute_scaled_error(y_true, y_pred):
  mae = tf.reduce_mean(tf.abs(y_true - y_pred))
  mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1])) 
  return mae / mae_naive_no_season

def evaluate_preds(y_true, y_pred):
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)
  mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
  mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
  rmse = tf.sqrt(mse)
  mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
  def mean_absolute_scaled_error(y_true, y_pred):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1])) 
    return mae / mae_naive_no_season
  mase = mean_absolute_scaled_error(y_true, y_pred)
  return {"mae": mae.numpy(),
          "mse": mse.numpy(),
          "rmse": rmse.numpy(),
          "mape": mape.numpy(),
          "mase": mase.numpy()}