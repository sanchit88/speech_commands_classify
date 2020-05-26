import numpy as np
import deepdish as dd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Input, Activation, Flatten, Dense
from tensorflow.keras.utils import to_categorical

#-----------------------------------------
#define parameters 
file_path_dataset = "/Users/sanchit/Documents/Projects/datasets/audio_data/speech_commands/audio_cleaned_dataset.h5"
batch_size = 64 
epochs = 20

#-----------------------------------------
def load_data(file_path="", class_mode="binary"):
    X = []
    y = []
    dataset = dd.io.load(file_path)
    
    # create class name to a label index map 
    list_classes = dataset.keys() 
    list_classes = sorted(list_classes)

    class_to_index = dict()
    for ind, class_name in enumerate(list_classes):
        class_to_index[class_name] = ind
        
    for class_name, list_feats in dataset.items():
        print(f"loading data for class: {class_name}")
        for feat in list_feats:
            data = img_to_array(feat)
            X.append(data)
            y.append(class_to_index[class_name])
        
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    
    if class_mode == "categorical":
        y = to_categorical(y)
        
    return X, y, class_to_index


#-----------------------------------------
def build_model(width, height, num_classes):
    """Build 2D CNN classifier."""
    input_data = Input(shape=(width, height, 1), name="input_layer")
    
    x = BatchNormalization(center=True, scale=True)(input_data) # normalize and scale the data first 
    x = Conv2D(16, kernel_size=(3, 7), strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(16, kernel_size=(7, 3), strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(32, kernel_size=(3, 3), strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(32, kernel_size=(3, 3), strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(64, kernel_size=(3, 3), strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = GlobalAveragePooling2D()(x)
    
    # classifier 
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    
    output = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=input_data, outputs=output)
    
    return model


#-----------------------------------------
# load the whole data 
X, y, class_to_ind_map = load_data(file_path=file_path_dataset, class_mode="categorical")
num_classes = len(class_to_ind_map)

#-----------------------------------------
# split the dataset into training and validation 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)
nb_train_samples = len(X_train)
nb_test_samples = len(X_test)


#-----------------------------------------
# create train and test generators 
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator() 

train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
test_generator = test_datagen.flow(X_test, y_test, batch_size=batch_size, shuffle=False)

#-----------------------------------------
# get the model and compile it 
model = build_model(128, 44, num_classes)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


#-----------------------------------------
# train the model     
H = model.fit_generator(train_generator, 
                        steps_per_epoch=nb_train_samples // batch_size, 
                        epochs=epochs, 
                        validation_data=test_generator, 
                        validation_steps=nb_test_samples // batch_size, 
                        workers = 4)

#-----------------------------------------
# plot train/val losses 


