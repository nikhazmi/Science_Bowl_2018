# Science_Bowl_2018
 
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
