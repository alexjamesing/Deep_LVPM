# import all necessary packages required for this tutorial
import tensorflow as tf
import numpy as np
from StructuralModel import StructuralModel
from tensorflow import keras
from keras import layers

# Generate random arrays using numpy

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train_cat), (x_test, y_test_cat) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train_cat, num_classes)
y_test = keras.utils.to_categorical(y_test_cat, num_classes)

data_train_list = [x_train, y_train]
data_test_list = [x_test, y_test]

MNIST_image_model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-5)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-5)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.5)

    ]
)

data_input = keras.Input(shape = 10)
MNIST_label_model=keras.Model(inputs=data_input,outputs=data_input)
  
# Here, we define a new adjacency matrix, which defines which data views to connect
Path = tf.constant([[0,1],
            [1,0]])

regularizer_list = [None,None] ## regularizer_list 

ndims = 9 # the number of DLVs we wish to extract
tot_num = x_train.shape[0] # the total number of samples, which is used for internal normalisation
batch_size = 32
epochs = 20

model_list = [MNIST_image_model, MNIST_label_model] 

DLVPM_Model = StructuralModel(Path, model_list, regularizer_list, tot_num, ndims, epochs, batch_size)

optimizer_list = [keras.optimizers.Adam(learning_rate=1e-4),keras.optimizers.Adam(learning_rate=1e-4)]

DLVPM_Model.compile(optimizer=optimizer_list)

DLVPM_Model.fit(data_train_list, batch_size=batch_size, epochs=epochs,verbose=True, validation_split=0.1)

metrics = DLVPM_Model.evaluate(data_test_list)

DLVs = DLVPM_Model.predict(data_test_list)

Cmat1 = np.corrcoef(DLVs[:,0,:].T)
Cmat2 = np.corrcoef(DLVs[:,:,0].T)

print(np.min(Cmat2-np.eye(Cmat2.shape[0])))

#DLVPM_Model.save('output_folder/DLVPM_Model.keras')

# import sklearn as sk
# from sklearn import svm
# from sklearn.multioutput import MultiOutputClassifier

## Here, we can run things, such as models trained on sklearn, for prediction of
## binary outputs


