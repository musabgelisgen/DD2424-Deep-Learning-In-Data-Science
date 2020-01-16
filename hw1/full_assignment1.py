from assignment import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import pickle

def main():
  np.random.seed(20)
  X_test, y_test, Y_test = load_data(batch_file = "cifar-10-batches-py/test_batch", number=None, number_f=None, k=10)
  X_train, y_train , Y_train = load_data(batch_file = "cifar-10-batches-py/data_batch_1", number=None, number_f=None, k=10)
  X_val, y_val , Y_val = load_data(batch_file = "cifar-10-batches-py/data_batch_2", number=None, number_f=None, k=10)

  neural_network = NeuralNetwork(X_train, Y_train)
  neural_network.fit(X_train, Y_train, X_val, Y_val)
  Y_pred_test = neural_network.softmax(np.dot(neural_network.w, X_test) + neural_network.b)
  accuracy = neural_network.compute_accuracy(Y_pred_test, Y_test)
  print(accuracy)


def load_data(batch_file, number, number_f, k):
  with open(batch_file, 'rb') as ph:
    data = pickle.load(ph, encoding='bytes')
    # preprocess
    X = data[b"data"]/255.0
    y = np.array(data[b"labels"])
    # one-hot representation in Y
    Y = np.zeros((y.shape[0], k))
    Y[np.arange(y.shape[0]), y] = 1

  if not(number):
    return X.T, y, Y.T
  else:
    return X[:number, :number_f].T, y[:number], Y[:number].T

if __name__ == "__main__":
  main()

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:

def __init__(self, X_train, Y_train, **kwargs):
  variables = {
      # network params
      "learning_rate":0.1,
      "lamda":0,
      "batch_size":100,
      "epoch_number":40,

      "h_parameter":1e-6,
      "mean_of_weights":0,
      "sigma_weights": 0.01,
      "cifar10_labels": ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
  }

  for var, default in variables.items():
      setattr(self, var, kwargs.get(var, default))

  # vector
  self.d = X_train.shape[0]
  self.n = X_train.shape[1]
  self.k = Y_train.shape[0]

  # weight and bias initialization
  self.w = np.random.normal(self.mean_of_weights, self.sigma_weights, (self.k, self.d))
  self.b = np.random.normal(self.mean_of_weights, self.sigma_weights, (self.k,1))


def fit(self, X_train, Y_train, X_validation, Y_validation):

  self.cost_hist_training = []
  self.cost_hist_validation = []
  self.acc_hist_training = []
  self.acc_hist_validation = []
  number_of_batches = int (self.n / self.batch_size)

  for i in range (self.epoch_number):
    for j in range (number_of_batches):
      j_init = j * self.batch_size
      j_final = self.batch_size + j*self.batch_size

      X_batch = X_train[:, j_init:j_final]
      Y_batch = Y_train[:, j_init:j_final]

      Y_pred = self.softmax( np.dot( self.w, X_batch) + self.b)
      w_grad, b_grad = self.compute_gradients(X_batch, Y_batch, Y_pred)

      updated_w = self.w - self.learning_rate * w_grad
      self.w = updated_w

      updated_b = self.b - self.learning_rate * b_grad
      self.b = updated_b

    self.report_performance(X_train, Y_train, X_validation, Y_validation, i)

  self.plot_cost()
  self.plot_weights()

def check_gradients(self, X, Y):

  w_grad_number = np.zeros((self.k, self.d))
  Y_pred = self.softmax(np.dot(self.w, X) + self.b)
  w_grad, b_grad = self.compute_gradients(X, Y, Y_pred)

  b_grad_number, w_grad_number = self.compute_gradient_anatically(X, Y)
  # b_grad_number, w_grad_number = self.compute_gradient_numerically(X, Y)

  w_grad_number_vec = w_grad_number.flatten()
  w_grad_vector = w_grad.flatten()
  x_w = np.arange(1, w_grad_vector.shape[0] + 1)

  b_grad_vec = b_grad.flatten()
  b_grad_number_vec = b_grad_number.flatten()
  x_b = np.arange(1, b_grad.shape[0] + 1)

  return x_w, w_grad_vector, x_b, b_grad_vec

def softmax(self, Y_predict):
  return np.exp(Y_predict) / np.dot(np.ones(Y_predict.shape[0]).T, np.exp(Y_predict))

def compute_cost(self, X, Y_actual):
  Y_pred = self.softmax(np.dot(self.w, X) + self.b)
  return self.cross_entropy(Y_actual, Y_pred) / X.shape[1] + self.lamda * np.sum(self.w ** 2)

def cross_entropy(self, Y_true, Y_pred):
  return np.sum(-np.log(np.sum(Y_true * Y_pred, axis=0)), axis=0)

def compute_accuracy(self,Y_predict, Y_actual):
  y_p = np.array(np.argmax(Y_predict, axis=0))
  y = np.array(np.argmax(Y_actual, axis=0))
  correct = len(np.where(np.array(np.argmax(Y_actual, axis=0))==np.array(np.argmax(Y_predict, axis=0)))[0])
  return correct/np.array(np.argmax(Y_actual, axis=0)).shape[0]

def compute_gradients(self, X_batch, y_true_batch, y_pred_batch):
  grad_loss_w = 1/self.batch_size * np.dot(y_pred_batch - y_true_batch, X_batch.T)
  grad_loss_b = 1/self.batch_size * np.sum(y_pred_batch - y_true_batch, axis=1).reshape(-1,1)
  return grad_loss_w + 2*self.lamda*self.w, grad_loss_b

def report_performance(self, X_train, Y_train, X_validation, Y_validation, epoch):
  Y_pred_train = self.softmax(np.dot(self.w, X_train) + self.b)
  Y_pred_validation = self.softmax(np.dot(self.w, X_validation) + self.b)

  self.cost_hist_training.append(self.compute_cost(X_train, Y_pred_train))
  self.acc_hist_training.append(self.compute_accuracy(Y_pred_train, Y_train))
  self.cost_hist_validation.append(self.compute_cost(X_validation, Y_pred_validation))
  self.acc_hist_validation.append(self.compute_accuracy(Y_pred_validation, Y_validation))

  print("Epoch: ", epoch, " - Accuracy: ", self.compute_accuracy(Y_pred_train, Y_train), " - Cost: ", self.compute_cost(X_train, Y_pred_train))

  # print("Epoch: ", epoch, " - Accuracy: ", self.compute_accuracy(Y_pred_validation, Y_validation), " - Cost: ", self.compute_cost(X_validation, Y_pred_validation))


def plot_cost(self):
  leng = len(self.cost_hist_training) + 1
  x = list(range(1, leng))
  plt.plot(x, self.cost_hist_training, label = "Training cost")
  plt.plot(x, self.cost_hist_validation, label = "Validation cost")
  plt.title("Training and validation cost")
  plt.xlabel("Epochs")
  plt.ylabel("Cost")
  plt.legend()
  plt.show()

def plot_accuracy(self):
  leng = len(self.cost_hist_training) + 1
  x = list(range(1, leng))
  plt.plot(x, self.acc_hist_training, label = "Training accuracy")
  plt.plot(x, self.acc_hist_validation, label = "Validation accuracy")
  plt.title("Training and validation accuracy")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.show()

def plot_weights(self):
  for i in range(self.k):
    w_image = self.w[i,:].reshape((32,32,3))
    plt.imshow((w_image*255).astype('uint8'))
    plt.title(self.cifar10_labels[i])
    plt.show()

def compute_gradient_numerically(self, X, Y):
  w_grad = np.zeros((self.k, self.d))
  b_grad = np.zeros((self.k,1))
  cost = self.compute_cost(X, Y)
  for i in range(self.b.shape[0]):
    self.b[i] = self.b[i] + self.h_parameter
    b_grad[i] = (self.compute_cost(X, Y)-cost) / self.h_parameter
    self.b[i] = self.b[i] - self.h_parameter
  for i in range(self.w.shape[0]):
    for j in range(self.w.shape[1]):
      self.w[i,j] = self.w[i,j] + self.h_parameter
      w_grad[i,j] = (self.compute_cost(X, Y)-cost) / self.h_parameter
      self.w[i,j] = self.w[i,j] -self.h_parameter
  return b_grad, w_grad

def compute_gradient_anatically(self, X, Y):
  w_grad = np.zeros((self.k, self.d))
  b_grad = np.zeros((self.k, 1))

  for i in range(self.b.shape[0]):
    self.b[i] = self.b[i] - self.h_parameter
    cost = self.compute_cost(X, Y)
    self.b[i] = self.b[i] + 2 * self.h_parameter
    b_grad[i] = (self.compute_cost(X, Y) - cost) / (2 * self.h_parameter)
    self.b[i] = self.b[i] - self.h_parameter
  for i in range(self.w.shape[0]):
    for j in range(self.w.shape[1]):
      self.w[i, j] = self.w[i, j] - self.h_parameter
      cost = self.compute_cost(X, Y)
      self.w[i, j] = self.w[i, j] + 2 * self.h_parameter
      w_grad[i, j] = (self.compute_cost(X, Y) - cost) / (2 * self.h_parameter)
      self.w[i, j] = self.w[i, j] - self.h_parameter
  return b_grad, w_grad
