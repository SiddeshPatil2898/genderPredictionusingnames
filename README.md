# Gender Prediction using names - LSTM model
# Gender Predicition using LSTM
The code trains a names classifier model using a dataset of Indian male and female names. It uses PyTorch and TorchText for the deep learning model and data preprocessing. Here are the main steps:

# Data Preparation:

The code reads male and female names from two CSV files using pandas.
The two DataFrames are concatenated to create a single DataFrame.
The DataFrame is saved to a new CSV file and then read back.
Any rows with missing values are dropped.
The order of rows in the DataFrame is shuffled randomly.
The dataset is split into training and test sets.
- `Dataset Class`:

The NamesDataset class is defined, which inherits from the torch.utils.data.Dataset class.
It takes the names DataFrame and character vocabulary as input during initialization.
The class provides methods to retrieve the names tensor and corresponding gender label for each sample in the dataset.

- `Model Definition`:

The NamesClassifier class is defined, which inherits from nn.Module.
It consists of an embedding layer, an LSTM layer, and two linear layers.
The model takes a name tensor as input and produces gender predictions.

-`Model Training`:

The code initializes an instance of the NamesClassifier model.
The model's predictions are computed for the training data to create initial predictions.
A confusion matrix is computed using the predicted and actual genders of the training data.
The cross-entropy loss function and Adam optimizer are defined.
The model is trained using a loop that iterates over the training dataset.
The loss is computed, gradients are backpropagated, and the optimizer updates the model's parameters.
After every 1000 steps, the total loss is printed.

-`Model Evaluation`:

The model's predictions are computed for the test data.
The test data's predictions are used to calculate a confusion matrix and accuracy score.
The model's predictions and actual genders are used to calculate another confusion matrix and accuracy score for the training data.
The code displays the test data rows where the predicted gender matches the actual gender.
# Description:
The code trains a names classifier model using PyTorch and TorchText. It uses an LSTM-based model to predict the gender of Indian names. The dataset is prepared by reading male and female names from CSV files, concatenating them, and performing data preprocessing. The model is trained using the training set and evaluated on the test set. Evaluation metrics such as confusion matrix and accuracy score are computed for both training and test data.
