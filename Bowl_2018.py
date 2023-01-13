#%%
#1. Import necessary packages
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import datetime

from IPython.display import clear_output

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras import callbacks

#%%
"""
    This is a script that demonstrates how to import necessary packages, download data, preprocess the data, and perform train-test split using TensorFlow library.

    1. The script starts by importing necessary packages such as pandas, numpy, opencv, matplotlib, tensorflow, and tensorflow_datasets.
    2. The data is then downloaded using the tensorflow_datasets module and stored in the root_path1 and root_path2 variables.
    3. The script then loads the images and masks using opencv, resizing them to 128x128, and appending them to the images and masks lists.
    4. The lists are then converted into numpy arrays and the mask dimension is expanded. The mask values are also converted from [0,255] into [0,1].
    5. The data is then split into train and test sets using the train_test_split function from sklearn.model_selection.
    6. Finally, the numpy arrays are converted into tensor slices and the images and masks are combined using the zip method to create train and test datasets.
    7. The script defines a class called Augment that is subclassed from keras.layers.Layer. The class applies data augmentation by randomly flipping the inputs and labels horizontally.
    8. The script then builds the dataset by applying caching, shuffling, batching, repeating, and data augmentation to the train_dataset. The test_dataset is also batched.
    9. The script includes a function to visualize some of the images in the dataset as an example.
    10. The script starts to develop the model by using a pretrained model (MobileNetV2) as the feature extractor. 
        The script uses the outputs from certain activation layers in the pretrained model as the outputs from the feature extractor. 
        Then, an instance of the feature extractor is created and the trainable property is set to false.
        10.4. The script defines the upsampling path using the pix2pix upsample function, which takes the number of output channels and the number of convolutional layers as arguments.
        10.5. The script uses the functional API to construct the entire U-net model by combining the feature extractor and upsampling path. 
              The final output of the model is a segmentation map with the same height and width as the input image, but with the number of channels equal to the number of output channels.
        10.6. The script uses the function unet(output_channels) to create the model. The number of output channels is passed as an argument (in this case, 3)
    11. The script compiles the model by specifying the optimizer and loss function. The loss function used is SparseCategoricalCrossentropy, which is appropriate for multi-class classification problems.
    12. The script includes two functions to show predictions. The first function, create_mask(pred_mask) takes the predicted mask as an input and converts it into an image format, 
        the second function show_predictions(dataset=None,num=1) takes an optional dataset and number of predictions to display. 
        It uses the first function to create a mask from the model predictions and display it along with the input image and true mask. 
        If no dataset is provided, the function will display the predictions on the sample image and mask that was displayed in the previous step.
    13. The script creates a callback function called DisplayCallback that inherits from the keras.callbacks.Callback class. The function overrides the on_epoch_end method to clear the output, call the show_predictions function, and print a message indicating the current epoch. 
        This callback function allows to see the predictions made by the model during training.
    14. The script then trains the model using the fit function, passing the train_batches as the training data, test_batches as the validation data, 
        and specifying the number of epochs, validation steps and steps per epoch. The script also pass the DisplayCallback and Tensorboard callback for monitoring the training progress.
    15. The script then uses the show_predictions function to display predictions made by the trained model on the test dataset.
    16.The script saves the trained model to the file "model.h5" in the current working directory.
"""
# %%
#2. Download the data using tensorflow dataset module
root_path1 = os.path.join(os.getcwd(), 'Datasets', 'train')
root_path2 = os.path.join(os.getcwd(), 'Datasets', 'test')
#%%
#2.2 Prepare empty list to hold the data
images = []
masks = []

#%%
#2.3. Load the images using opencv
image_dir = os.path.join(root_path1,'inputs')
for image_file in os.listdir(image_dir):
    img = cv2.imread(os.path.join(image_dir,image_file))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(128,128))
    images.append(img)


#%%
#2.4. Load the masks
masks_dir = os.path.join(root_path1,'masks')
for mask_file in os.listdir(masks_dir):
    mask = cv2.imread(os.path.join(masks_dir,mask_file),cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128,128))
    masks.append(mask)

#%%
#2.5. Convert the list of np array into a np array
images_np = np.array(images)
masks_np = np.array(masks)

#%%
#3. Data preprocessing
#3.1. Expand the mask dimension
masks_np_exp = np.expand_dims(masks_np,axis=-1)
#Check the mask output
print(np.unique(masks_np_exp[0]))

#%%
#3.2. Convert the mask values from [0,255] into [0,1]
converted_masks = np.round(masks_np_exp / 255.0).astype(np.int64)

#Check the mask output
print(np.unique(converted_masks[0]))

#%%
#3.3. Normalize the images
converted_images = images_np/ 255.0
#%%
#4. Perform train-test split
from sklearn.model_selection import train_test_split

SEED = 42
X_train, X_test, y_train, y_test = train_test_split(converted_images, converted_masks, test_size=0.2, random_state=SEED)

#%%
#5. Convert the numpy arrays into tensor slices
X_train_tensor = tf.data.Dataset.from_tensor_slices(X_train)
X_test_tensor = tf.data.Dataset.from_tensor_slices(X_test)
y_train_tensor = tf.data.Dataset.from_tensor_slices(y_train)
y_test_tensor = tf.data.Dataset.from_tensor_slices(y_test)

#%%
#6. Combine the images and masks using the zip method
train_dataset = tf.data.Dataset.zip((X_train_tensor,y_train_tensor))
test_dataset = tf.data.Dataset.zip((X_test_tensor,y_test_tensor))

#%%
#7. Define data augmentation pipeline as a single layer through subclassing
class Augment(keras.layers.Layer):
    def __init__(self,seed=42):
        super().__init__()
        self.augment_inputs = keras.layers.RandomFlip(mode='horizontal',seed=seed)
        self.augment_labels = keras.layers.RandomFlip(mode='horizontal',seed=seed)

    def call(self,inputs,labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels

#%%
#8. Build the dataset
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
TRAIN_SIZE = len(train_dataset)
STEP_PER_EPOCH = TRAIN_SIZE//BATCH_SIZE

train_batches = (
    train_dataset
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
#%%
test_batches = test_dataset.batch(BATCH_SIZE)

# %%
#9. Visualize some pictures as example
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image','True Mask','Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
    plt.show()

for images,masks in train_batches.take(2):
    sample_image,sample_mask = images[0],masks[0]
    display([sample_image,sample_mask])

# %%
#10. Model development
#10.1. Use a pretrained model as the feature extractor
base_model = keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)
base_model.summary()

#10.2. Use these activation layers as the outputs from the feature extractor (some of these outputs will be used to perform concatenation at the upsampling path)
layer_names = [
    'block_1_expand_relu',      #64x64
    'block_3_expand_relu',      #32x32
    'block_6_expand_relu',      #16x16
    'block_13_expand_relu',     #8x8
    'block_16_project'          #4x4
    ]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

#10.3. Instantiate the feature extractor
down_stack = keras.Model(inputs=base_model.input,outputs=base_model_outputs)
down_stack.trainable = False

#10.4. Define the upsampling path
up_stack = [
    pix2pix.upsample(512,3),    #4x4 --> 8x8
    pix2pix.upsample(256,3),    #8x8 --> 16x16
    pix2pix.upsample(128,3),    #16x16 --> 32x32
    pix2pix.upsample(64,3)      #32x32 --> 64x64
]

#10.5. Use functional API to construct the entire U-net
def unet(output_channels:int):
    inputs = keras.layers.Input(shape=[128,128,3])
    #Downsample through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    #Build the upsampling path and establish the concatenation
    for up, skip in zip(up_stack,skips):
        x = up(x)
        concat = keras.layers.Concatenate()
        x = concat([x,skip])
    
    #Use a transpose convolution layer to perform the last upsampling, this will become the output layer
    last = keras.layers.Conv2DTranspose(filters=output_channels,kernel_size=3,  strides=2,padding='same') #64x64 --> 128x128
    outputs = last(x)
    model = keras.Model(inputs=inputs,outputs=outputs)
    return model

# %%
#10.6. Use the function to create the model
OUTPUT_CHANNELS = 3
model = unet(OUTPUT_CHANNELS)
model.summary()
keras.utils.plot_model(model)

# %%
#11. Compile the model
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])

# %%
#12. Create functions to show predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])
    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])

show_predictions()
# %%
#13. Create a callback function to make use of the show_predictions function
class DisplayCallback(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample prediction after epoch {}\n'.format(epoch+1))
log_path = os.path.join('log_dir', 'tl', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = callbacks.TensorBoard(log_dir = log_path)
# %%
#14. Model training
EPOCHS = 5
VAL_SUBSPLITS = 5
VAL_STEPS = 200
history = model.fit(train_batches,validation_data=test_batches,epochs=EPOCHS,   validation_steps = VAL_STEPS, steps_per_epoch=STEP_PER_EPOCH,callbacks=[DisplayCallback(), tb])
# %%
#15. Model deployment
show_predictions(test_batches,3)

#%% 
#16. Save the model
model.save('model.h5')
# %%
