# import all necessary packages required for this tutorial
import tensorflow as tf
import numpy as np
import deep_lvpm
from tensorflow import keras
from keras import layers
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from deep_lvpm.models.Siamese import Siamese ## Here, we import the main StructuralModel class used in deep-lvpm
from deep_lvpm.models.SiameseVicReg import SiameseVicReg ## Here, we import the main StructuralModel class used in deep-lvpm
from deep_lvpm.models.StructuralModel import StructuralModel ## Here, we import the main StructuralModel class used in deep-lvpm
from deep_lvpm.models.SiameseBarlow import SiameseBarlow ## Here, we import the main StructuralModel class used in deep-lvpm

#tf.config.run_functions_eagerly(True)

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train_cat), (x_test, y_test_cat) = keras.datasets.mnist.load_data()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Reshape data to add channel dimension (1, for grayscale)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Create an image data generator for augmentation
datagen = ImageDataGenerator(
    rotation_range=50,
    zoom_range=0.3,
    width_shift_range=0.5,
    height_shift_range=0.5)

# Normalizing dataset
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


def pair_generator(x_train, batch_size):
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1)
    while True:
        idx = np.random.choice(np.arange(len(x_train)), batch_size)
        batch = x_train[idx]
        batch_1 = datagen.flow(batch, batch_size=batch_size, shuffle=False).next()
        batch_2 = datagen.flow(batch, batch_size=batch_size, shuffle=False).next()
        
       
        yield ([[batch_1, batch_2]],)


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train_cat, num_classes)
y_test = keras.utils.to_categorical(y_test_cat, num_classes)

data_train_list = [x_train, y_train]
data_test_list = [x_test, y_test]

MNIST_image_model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        #layers.BatchNormalization(),
        layers.Dense(500)

        #layers.Dense(100),
        #layers.BatchNormalization(),
        #layers.Dropout(0.8)

    ]
)

# data_input = keras.Input(shape = 10)
# MNIST_label_model=keras.Model(inputs=data_input,outputs=data_input)
  
model_list = [MNIST_image_model, MNIST_image_model] 

# # Here, we define a new adjacency matrix, which defines which data views to connect
Path = tf.constant([[0,1],
             [1,0]])

regularizer_list = [keras.regularizers.l1_l2(l1=1e-4, l2=1e-4),keras.regularizers.l1_l2(l1=1e-4, l2=1e-4)] ## regularizer_list 

ndims = 500 # the number of DLVs we wish to extract
tot_num = x_train.shape[0] # the total number of samples, which is used for internal normalisation
#tot_num = 100
batch_size = 32
epochs = 1

DLVPM_Model = Siamese(Path, model_list, regularizer_list, tot_num, ndims, momentum=0.8)
#DLVPM_Model = SiameseVicReg(Path, model_list, regularizer_list, tot_num, ndims)
#DLVPM_Model = SiameseBarlow(Path, model_list, regularizer_list, tot_num, ndims)

optimizer_list = [keras.optimizers.Adam(learning_rate=1e-5),keras.optimizers.Adam(learning_rate=1e-5)]

DLVPM_Model.compile(optimizer=optimizer_list)

DLVPM_Model.fit(pair_generator(x_train, batch_size=2048), steps_per_epoch = 400,epochs=1)

image_model = DLVPM_Model.model_list[0]

DLVs_train = image_model.predict(x_train)

DLVs_test = image_model.predict(x_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define a simple Sequential model for classification
model = Sequential([
    Dense(num_classes, activation='sigmoid',input_shape=(DLVs_train.shape[1],))  # Using sigmoid for binary classification
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(DLVs_train, y_train, validation_data=(DLVs_test, y_test), epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(DLVs_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')

# #DLVPM_Model.save('output_folder/DLVPM_Model.keras')

# image_DLVs = DLVPM_Model.model_list[0].predict(data_test_list[0])

# ## Here, we randomy select 100 examples for plotting
# random_indices = np.random.choice(image_DLVs.shape[0], size=100, replace=False)

# image_DLVs_plot = image_DLVs[random_indices,:]
# y_test_plot = y_test[random_indices,:]

# # Apply t-SNE
# tsne = TSNE(n_components=2, random_state=42)
# tsne_results = tsne.fit_transform(image_DLVs_plot)

# # Plot
# plt.figure(figsize=(12, 8))

# for i in range(y_test_plot.shape[1]):
#     points = tsne_results[y_test_plot[:, i] == 1]
#     plt.scatter(points[:, 0], points[:, 1], label=f'Category {i+1}')

# plt.title('t-SNE projection of the dataset')
# plt.legend()
# plt.savefig('/Users/ing/Downloads/figure_out.png')
# plt.show()




