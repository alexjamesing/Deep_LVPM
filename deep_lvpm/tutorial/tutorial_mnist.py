
# import all necessary packages required for this tutorial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from deep_lvpm.model import StructuralModel ## Here, we import the main StructuralModel class used in deep-lvpm

# Model / data parameters
num_classes = 10
input_shape = (1, 28, 28)

# Load the data and split it between train and test sets
train_dataset = datasets.MNIST(root="data", train=True, download=True)
test_dataset = datasets.MNIST(root="data", train=False, download=True)
x_train = train_dataset.data.numpy()
y_train_cat = np.asarray(train_dataset.targets, dtype="int64")
x_test = test_dataset.data.numpy()
y_test_cat = np.asarray(test_dataset.targets, dtype="int64")

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (n, 1, 28, 28)
x_train = np.expand_dims(x_train, 1)
x_test = np.expand_dims(x_test, 1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = np.eye(num_classes, dtype="float32")[y_train_cat]
y_test = np.eye(num_classes, dtype="float32")[y_test_cat]


data_train_list = [x_train, y_train]
data_test_list = [x_test, y_test]


class MNISTImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(64 * 5 * 5, 512)
        self.dropout = nn.Dropout(0.1)
        self.n_inputs = 1

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = self.flatten(x)
        x = F.relu(self.dense(x))
        x = self.dropout(x)
        return x


class MNISTLabelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_inputs = 1

    def forward(self, inputs):
        return inputs


MNIST_image_model = MNISTImageModel()
MNIST_label_model = MNISTLabelModel()


# Define a model list, which will then be used as an input to the DLVPM model
model_list = [MNIST_image_model, MNIST_label_model]

# Here, we define a new adjacency matrix, which defines which data views to connect
Path = np.array([[0,1],
            [1,0]], dtype="float32")

regularizer_list = [None,None] ## regularizer_list

ndims = 9 # the number of DLVs we wish to extract
tot_num = x_train.shape[0] # the total number of samples, which is used for internal normalisation
batch_size = 256
epochs = 20

DLVPM_Model = StructuralModel(Path, model_list, regularizer_list, tot_num, ndims, orthogonalization="zca", train_DLV=False)

optimizer_list = [torch.optim.Adam(model.parameters(), lr=1e-5) for model in DLVPM_Model.model_list]

DLVPM_Model.compile(optimizer=optimizer_list)

DLVPM_Model.fit(data_train_list, batch_size=batch_size, epochs=epochs,verbose=True, validation_split=0.1)

metrics = DLVPM_Model.evaluate(data_test_list)

DLVs = DLVPM_Model.predict(data_test_list)

Cmat1 = np.corrcoef(DLVs[:,0,:].T)

image_DLVs = DLVPM_Model.model_list[0](torch.as_tensor(data_test_list[0], dtype=torch.float32)).detach().cpu().numpy()

## Here, we randomy select 100 examples for plotting
random_indices = np.random.choice(image_DLVs.shape[0], size=100, replace=False)

image_DLVs_plot = image_DLVs[random_indices,:]
y_test_plot = y_test[random_indices,:]

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(image_DLVs_plot)

# Plot
plt.figure(figsize=(12, 8))

for i in range(y_test_plot.shape[1]):
    points = tsne_results[y_test_plot[:, i] == 1]
    plt.scatter(points[:, 0], points[:, 1], label=f'Category {i+1}')

plt.title('t-SNE projection of the dataset')
plt.legend()
