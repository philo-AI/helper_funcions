import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np
import sklearn 
from PIL import Image
from PIL import ImageOps
import tempfile
import six
from tensorflow import keras as ks
from six import BytesIO
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
def preproce_time_series(data,index,tts):
  input_data = data[index]
  targets = data[index]
  dataset = tf.keras.utils.timeseries_dataset_from_array(
      input_data, targets, sequence_length=10)
  for batch in dataset:
    inputs, targets = batch
    assert np.array_equal(inputs[0], data[index])
    assert np.array_equal(targets[0], data[index])
    break
  split_size = int(tts * len(inputs))
  x_train, y_train = inputs[:split_size], targets[:split_size]
  x_test, y_test = inputs[split_size:], targets[split_size:]
  print(f"this function turns data to a ready to train preproccesed stuff x_train = {x_train.shape}, y_train = {y_train.shape}, x_test= {x_test.shape}, y_test = {y_test.shape}")
  print(f"FYI: X is the windows while Y is the horizon")
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
  def mean_absolute_scaled_error(y_trues, y_preds):
    mae = tf.reduce_mean(tf.abs(y_trues - y_preds))
    mae_naive_no_season = tf.reduce_mean(tf.abs(y_trues[1:] - y_trues[:-1])) 
    return mae / mae_naive_no_season
  mase = mean_absolute_scaled_error(y_trues=y_true, y_preds=y_pred)
  return {"mae": mae.numpy(),
          "mse": mse.numpy(),
          "rmse": rmse.numpy(),
          "mape": mape.numpy(),
          "mase": mase.numpy()}
  
def download_and_resize_image(url, new_width, new_height):  
    # create a temporary file ending with ".jpg"
    _, filename = tempfile.mkstemp(suffix=".jpg")
    
    # opens the given URL
    response = six.moves.urllib.request.urlopen(url)
    
    # reads the image fetched from the URL
    image_data = response.read()
    
    # puts the image data in memory buffer
    image_data = BytesIO(image_data)
    
    # opens the image
    pil_image = Image.open(image_data)
    
    # resizes the image. will crop if aspect ratio is different.
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    
    # converts to the RGB colorspace
    pil_image_rgb = pil_image.convert("RGB")
    
    # saves the image to the temporary file created earlier
    pil_image_rgb.save(filename, format="JPEG", quality=90)
    
    print("Image downloaded to %s." % filename)
    
    return filename

def load_img(path):
    '''
    Loads a JPEG image and converts it to a tensor.
    
    Args:
        path (string) -- path to a locally saved JPEG image
    
    Returns:
        (tensor) -- an image tensor
    '''
    
    # read the file
    img = tf.io.read_file(path)
    
    # convert to a tensor
    img = tf.image.decode_jpeg(img, channels=3)
    
    return img


def run_detector(detector, path):
    '''
    Runs inference on a local file using an object detection model.
    
    Args:
        detector (model) -- an object detection model loaded from TF Hub
        path (string) -- path to an image saved locally
    '''
    
    # load an image tensor from a local file path
    img = load_img(path)

    # add a batch dimension in front of the tensor
    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    
    # run inference using the model
    result = detector(converted_img)

    # save the results in a dictionary
    result = {key:value.numpy() for key,value in result.items()}

    # print results
    print("Found %d objects." % len(result["detection_scores"]))

    print(result["detection_scores"])
    print(result["detection_class_entities"])
    print(result["detection_boxes"])

def training_model_for_GANs(genrator_optimezer, detector_optimizer, gen_loss, detector_loss,generator, detector,training_data,epochs,input_shape,Callbacks=None,val_data=None):
  class GAN_MODEL(ks.models.Model): 
      def __init__(self, generator, discriminator, *args, **kwargs):
          # Pass through args and kwargs to base class 
          super().__init__(*args, **kwargs)
          
          # Create attributes for gen and disc
          self.generator = generator 
          self.discriminator =  detector
          
      def compile(self, genrator_optimezer, detector_optimizer, gen_loss, detector_loss, *args, **kwargs): 
          # Compile with base class
          super().compile(*args, **kwargs)
          
          # Create attributes for losses and optimizers
          self.g_opt = genrator_optimezer
          self.d_opt = detector_optimizer
          self.g_loss = gen_loss
          self.d_loss = detector_loss

      def train_step(self, batch):
          # Get the data 
          real_images = batch
          fake_images = self.generator(tf.random.normal((input_shape)), training=False)
          
          # Train the discriminator
          with tf.GradientTape() as d_tape: 
              # Pass the real and fake images to the discriminator model
              yhat_real = self.discriminator(real_images, training=True) 
              yhat_fake = self.discriminator(fake_images, training=True)
              yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)
              
              # Create labels for real and fakes images
              y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)
              
              # Add some noise to the TRUE outputs
              noise_real = 0.15*tf.random.uniform(tf.shape(yhat_real))
              noise_fake = -0.15*tf.random.uniform(tf.shape(yhat_fake))
              y_realfake += tf.concat([noise_real, noise_fake], axis=0)
              
              # Calculate loss - BINARYCROSS 
              total_d_loss = self.d_loss(y_realfake, yhat_realfake)
              
          # Apply backpropagation - nn learn 
          dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables) 
          self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))
          
          # Train the generator 
          with tf.GradientTape() as g_tape: 
              # Generate some new images
              gen_images = self.generator(tf.random.normal((input_shape)), training=True)
                                          
              # Create the predicted labels
              predicted_labels = self.discriminator(gen_images, training=False)
                                          
              # Calculate loss - trick to training to fake out the discriminator
              total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels) 
              
          # Apply backprop
          ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
          self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))
          
          return {"detector_loss":total_d_loss, "generator_loss":total_g_loss}
  fashgan = GAN_MODEL(generator, detector)
  fashgan.compile(genrator_optimezer, detector_optimizer, gen_loss, detector_loss)
  ## read before runing IF there are labels to your code (EG you loaded it from git hub or kaggle)
  ## use this training_data = training_data.as_numpy_iterator().next()
  ## and when your fitting use this
  ## fashgan.fit(training_data[0].....)
  if Callbacks == None and val_data==  None:
    return fashgan.fit(training_data,epochs=epochs)
  elif Callbacks == None and val_data != None:
    return fashgan.fit(training_data,epochs=epochs,validation_data=val_data)
  elif Callbacks != None and val_data == None:
    return fashgan.fit(training_data,epochs=epochs,callbacks=[Callbacks])
  else:
    return fashgan.fit(training_data,epochs=epochs,callbacks=[Callbacks],validation_data=val_data)
  

def gan_preproccesing(data):
  def scale_images(data): 
    image = data['image']
    return image / 255
  # Running the dataset through the scale_images preprocessing step
  ds = data.map(scale_images) 
  # Cache the dataset for that batch 
  ds = data.cache()
  # Shuffle it up 
  ds = data.shuffle(60000)
  # Batch into 128 images per sample
  ds = data.batch(128)
  # Reduces the likelihood of bottlenecking 
  ds = data.prefetch(64)
  return ds