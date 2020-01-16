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

  actual_h = np.maximum(0, np.dot(neural_network.w1, X_test) + neural_network.b1)
  Y_pred = neural_network.softmax(np.dot(neural_network.w2, actual_h) + neural_network.b2)

  accuracy = neural_network.compute_accuracy(Y_pred, Y_test)
  print(accuracy)


def load_data(batch_file, number, number_f, k):
  with open(batch_file, 'rb') as ph:
    data = pickle.load(ph, encoding='bytes')
    # preprocess
    var = data[b"data"]/255.0
    mean_vector = np.mean(var, axis = 0)
    std_vector = np.std(var, axis =0)
    X = (var - mean_vector) / std_vector

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

def main():

  X_train, y_train, Y_train, X_val, y_val, Y_val = get_all_data()
  X_test, y_test, Y_test = load_data(batch_file = "cifar-10-batches-py/test_batch", number=None, number_f=None, k=10)

  course_search = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
  fine_search = [1e-5, 2*1e-5, 3*1e-5, 4*1e-5, 5*1e-5, 6*1e-5, 7*1e-5, 8*1e-5, 9*1e-5, 1e-4, 2*1e-4, 3*1e-4, 4*1e-4, 5*1e-4, 6*1e-4, 7*1e-4, 8*1e-4, 9*1e-4, 1e-3]
  final_lamda = 0.0004
  # for lamda in course_search:
  for lamda in fine_search:
    params = {
      "lamda": lamda,
      "epochs": 16
    }
    neural_network = NeuralNetwork(X_train, Y_train, **params)
    neural_network.fit(X_train, Y_train, X_val, Y_val)
    actual_h = np.maximum(0, np.dot(neural_network.w1, X_val) + neural_network.b1)
    Y_pred = neural_network.softmax(np.dot(neural_network.w2, actual_h) + neural_network.b2)
    accuracy = neural_network.compute_accuracy(Y_pred, Y_val)
    print("\n Lambda: ", lamda, "\n Accuracy: ", accuracy)

  params = {
    "lamda": final_lamda,
    "epochs": 16
  }
  neural_network = NeuralNetwork(X_train, Y_train, **params)
  neural_network.fit(X_train, Y_train, X_val, Y_val)
  actual_h = np.maximum(0, np.dot(neural_network.w1, X_test) + neural_network.b1)
  Y_pred = neural_network.softmax(np.dot(neural_network.w2, actual_h) + neural_network.b2)

  accuracy = neural_network.compute_accuracy(Y_pred, Y_test)
  print(accuracy, " ", final_lamda)

def get_all_data():
  X1, y1, Y1 = load_data(batch_file="cifar-10-batches-py/data_batch_1", number=None, number_f=None, k=10)
  X2, y2, Y2 = load_data(batch_file="cifar-10-batches-py/data_batch_2", number=None, number_f=None, k=10)
  X3, y3, Y3 = load_data(batch_file="cifar-10-batches-py/data_batch_3", number=None, number_f=None, k=10)
  X4, y4, Y4 = load_data(batch_file="cifar-10-batches-py/data_batch_4", number=None, number_f=None, k=10)
  X5, y5, Y5 = load_data(batch_file="cifar-10-batches-py/data_batch_5", number=None, number_f=None, k=10)
  X_all = np.concatenate((X1, X2, X3, X4, X5), axis=1)
  y_all = np.concatenate((y1, y2, y3, y4, y5))
  Y_all = np.concatenate((Y1, Y2, Y3, Y4, Y5), axis=1)
  X_train = X_all[:, :-500]
  y_train = y_all[:-500]
  Y_train = Y_all[:, :-500]
  X_val = X_all[:, -500:]
  y_val = y_all[-500:]
  Y_val = Y_all[:, -500:]
  return X_train, y_train, Y_train, X_val, y_val, Y_val

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:

  def __init__(self, X_train, Y_train, **kwargs):
    variables = {
        # network params
        "learning_rate":1e-5,
        "lamda":8e-4,
        "batch_size":100,
        "epoch_number":10,

        "h_parameter":1e-6,
        "mean_of_weights":0,
        # "sigma_weights": 0.01,
        "cifar10_labels": ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"],
        #newly added
        "sigma_weights": "default",
        "h_size": 50,
        "learning_rate_min": 1e-5,
        "learning_rate_max": 1e-1
    }

    for var, default in variables.items():
        setattr(self, var, kwargs.get(var, default))

    # vector
    self.d = X_train.shape[0]
    self.n = X_train.shape[1]
    self.k = Y_train.shape[0]
    self.m = self.h_size

    # weight and bias initialization
    if self.sigma_weights == "default":
        sigma_weights_w1 = 1/np.sqrt(self.d)
        sigma_weights_w2 = 1/np.sqrt(self.m)
    elif type(sigma_weights == 'float'):
        sigma_weights_w1 = self.sigma_weights
        sigma_weights_w2 = self.sigma_weights

    self.w1 = np.random.normal(self.mean_of_weights, sigma_weights_w1, (self.m, self.d))
    self.w2 = np.random.normal(self.mean_of_weights, sigma_weights_w2, (self.k, self.m))

    self.b1 = np.zeros((self.m, 1))
    self.b2 = np.zeros((self.k, 1))
    self.ns = 2 * int(self.n / self.batch_size)


  def fit(self, X_train, Y_train, X_validation, Y_validation):

    self.cost_hist_training = []
    self.cost_hist_validation = []
    self.acc_hist_training = []
    self.acc_hist_validation = []
    number_of_batches = int (self.n / self.batch_size)

    for i in range(self.epoch_number):
      for j in range(number_of_batches):
        j_init = j * self.batch_size
        j_final = self.batch_size + j*self.batch_size

        X_batch = X_train[:, j_init:j_final]
        Y_batch = Y_train[:, j_init:j_final]

        actual_h = np.maximum(0, np.dot(self.w1, X_batch) + self.b1)
        Y_pred = self.softmax(np.dot(self.w2, actual_h) + self.b2)

        w1_grad, w2_grad, b1_grad, b2_grad = self.compute_gradients(X_batch, Y_batch, Y_pred, actual_h)

        self.w1 = self.w1 - self.learning_rate * w1_grad
        self.b1 = self.b1 - self.learning_rate * b1_grad
        self.w2 = self.w2 - self.learning_rate * w2_grad
        self.b2 = self.b2 - self.learning_rate * b2_grad
        self.learning_rate = self.cyclic_learning_rate(i * number_of_batches + j)
      self.report_performance(X_train, Y_train, X_validation, Y_validation, i)
    self.plot_cost()
    self.plot_accuracy()

  def check_gradients(self, X, Y):

    w_grad_number = np.zeros((self.k, self.d))
    actual_h = np.maximum(0, np.dot(self.w1, X) + self.b1)
    Y_pred = self.softmax(np.dot(self.w2, actual_h) + self.b2)
    w1_grad, w2_grad, b1_grad, b2_grad = self.compute_gradients(X, Y, Y_pred, actual_h)

    b1_grad_number, b2_grad_number, w1_grad_number, w2_grad_number = self.compute_gradient_anatically(X, Y)
    # b1_grad_number, b2_grad_number, w1_grad_number, w2_grad_number = self.compute_gradient_numerically(X, Y)

    w1_grad_number_vec = w1_grad_number.flatten()
    w1_grad_vector = w1_grad.flatten()
    x_w1 = np.arange(1, w1_grad_vector.shape[0] + 1)

    b1_grad_vec = b1_grad.flatten()
    b1_grad_number_vec = b1_grad_number.flatten()
    x_b1 = np.arange(1, b1_grad.shape[0] + 1)

    w2_grad_number_vec = w2_grad_number.flatten()
    w2_grad_vector = w2_grad.flatten()
    x_w2 = np.arange(1, w2_grad_vector.shape[0] + 1)

    b2_grad_vec = b2_grad.flatten()
    b2_grad_number_vec = b2_grad_number.flatten()
    x_b2 = np.arange(1, b2_grad.shape[0] + 1)

    return x_w1, w1_grad_vector, x_b1, b1_grad_vector
    # return x_w2, w2_grad_vector, x_b2, b2_grad_vector

  def softmax(self, Y_predict):
    return np.exp(Y_predict) / np.dot(np.ones(Y_predict.shape[0]).T, np.exp(Y_predict))

  def compute_cost(self, X, Y_actual):
    actual_h = np.maximum(0, np.dot(self.w1, X) + self.b1)
    Y_pred = self.softmax(np.dot(self.w2, actual_h) + self.b2)

    return self.cross_entropy(Y_actual, Y_pred) / X.shape[1] + (self.lamda * np.sum(self.w1**2) +  self.lamda * np.sum(self.w2**2))

  def cross_entropy(self, Y_true, Y_pred):
    return np.sum(-np.log(np.sum(Y_true * Y_pred, axis=0)), axis=0)

  def compute_accuracy(self,Y_predict, Y_actual):
    correct = len(np.where(np.array(np.argmax(Y_actual, axis=0))==np.array(np.argmax(Y_predict, axis=0)))[0])
    return correct/np.array(np.argmax(Y_actual, axis=0)).shape[0]

  def compute_gradients(self, X_batch, y_true_batch, y_pred_batch, actual_h):
    b2_grad = 1 / self.batch_size * np.sum(y_pred_batch - y_true_batch, axis=1).reshape(-1, 1)
    w2_grad = 1 / self.batch_size * np.dot(y_pred_batch - y_true_batch, actual_h.T)
    grad_batch = np.dot(self.w2.T, y_pred_batch - y_true_batch)

    actual_h_2 = np.zeros(actual_h.shape)
    range1 = actual_h.shape[0]
    range2 = actual_h.shape[1]
    for i in range(range1):
      for j in range(range2):
        if actual_h[i,j] > 0:
          actual_h_2[i, j] = 1
    grad_batch = grad_batch * actual_h_2

    w1_grad = 1 / self.batch_size * np.dot(grad_batch, X_batch.T)
    b1_grad = 1 / self.batch_size * np.sum(grad_batch, axis=1).reshape(-1, 1)

    return w1_grad + 2 * self.lamda * self.w1, w2_grad + 2 * self.lamda * self.w2, b1_grad, b2_grad


  def report_performance(self, X_train, Y_train, X_validation, Y_validation, epoch):
    Y_pred_train = self.softmax(np.dot(self.w2, np.maximum(0, np.dot(self.w1, X_train) + self.b1)) + self.b2)

    Y_pred_validation = self.softmax(np.dot(self.w2, np.maximum(0, np.dot(self.w1, X_validation) + self.b1)) + self.b2)

    self.cost_hist_training.append(self.compute_cost(X_train, Y_pred_train))
    self.acc_hist_training.append(self.compute_accuracy(Y_pred_train, Y_train))
    self.cost_hist_validation.append(self.compute_cost(X_validation, Y_pred_validation))
    self.acc_hist_validation.append(self.compute_accuracy(Y_pred_validation, Y_validation))

    print("Epoch: ", epoch, " - Accuracy: ", self.compute_accuracy(Y_pred_train, Y_train), " - Cost: ", self.compute_cost(X_train, Y_pred_train))

    # print("Epoch: ", epoch, " - Accuracy: ", self.compute_accuracy(Y_pred_validation, Y_validation), " - Cost: ", self.compute_cost(X_validation, Y_pred_validation))


  def plot_cost(self):
    leng = len(self.cost_hist_training) + 1
    x = list(range(1, leng))
    plt.plot(x, self.cost_hist_training, label = "Training loss")
    plt.plot(x, self.cost_hist_validation, label = "Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
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
      w_image = self.w1[i, :].reshape((32, 32, 3), order='F')
      plt.imshow(np.rot90(((w_image - w_image.min()) / (w_image.max() - w_image.min())), 3))
      plt.title(self.cifar10_labels[i])
      plt.show()

  def compute_gradient_numerically(self, X, Y):
    w1_grad = np.zeros((self.m, self.d))
    w2_grad = np.zeros((self.k, self.m))
    b1_grad = np.zeros((self.m, 1))
    b2_grad = np.zeros((self.k, 1))

    cost = self.compute_cost(X, Y)

    for i in range(self.b1.shape[0]):
      self.b1[i] = self.b1[i] + self.h_parameter
      b1_grad[i] = (self.compute_cost(X, Y)-cost) / self.h_parameter
      self.b1[i] = self.b1[i] - self.h_parameter

    for i in range(self.b2.shape[0]):
      self.b2[i] = self.b2[i] + self.h_parameter
      b2_grad[i] = (self.compute_cost(X, Y)-cost) / self.h_parameter
      self.b2[i] = self.b2[i] - self.h_parameter

    for i in range(self.w1.shape[0]):
      for j in range(self.w1.shape[1]):
        self.w1[i,j] = self.w1[i,j] + self.h_parameter
        w1_grad[i,j] = (self.compute_cost(X, Y)-cost) / self.h_parameter
        self.w1[i,j] = self.w1[i,j] - self.h_parameter

    for i in range(self.w2.shape[0]):
      for j in range(self.w2.shape[1]):
        self.w2[i,j] = self.w2[i,j] + self.h_parameter
        w2_grad[i,j] = (self.compute_cost(X, Y)-cost) / self.h_parameter
        self.w2[i,j] = self.w2[i,j] - self.h_parameter
    return b1_grad, b2_grad, w1_grad, w2_grad

  def compute_gradient_anatically(self, X, Y):
    w1_grad = np.zeros((self.m, self.d))
    w2_grad = np.zeros((self.k, self.m))
    b1_grad = np.zeros((self.m, 1))
    b2_grad = np.zeros((self.k, 1))

    for i in range(self.b1.shape[0]):
      self.b1[i] = self.b1[i] - self.h_parameter
      cost = self.compute_cost(X, Y)
      self.b1[i] = self.b1[i] + 2 * self.h_parameter
      b1_grad[i] = (self.compute_cost(X, Y) - cost) / (2 * self.h_parameter)
      self.b1[i] = self.b1[i] - self.h_parameter

    for i in range(self.b2.shape[0]):
      self.b2[i] = self.b2[i] - self.h_parameter
      cost = self.compute_cost(X, Y)
      self.b2[i] = self.b2[i] + 2 * self.h_parameter
      b2_grad[i] = (self.compute_cost(X, Y) - cost) / (2 * self.h_parameter)
      self.b2[i] = self.b2[i] - self.h_parameter

    for i in range(self.w1.shape[0]):
      for j in range(self.w1.shape[1]):
        self.w1[i, j] = self.w1[i, j] - self.h_parameter
        cost = self.compute_cost(X, Y)
        self.w1[i, j] = self.w1[i, j] + 2 * self.h_parameter
        w1_grad[i, j] = (self.compute_cost(X, Y) - cost) / (2 * self.h_parameter)
        self.w1[i, j] = self.w1[i, j] - self.h_parameter

    for i in range(self.w2.shape[0]):
      for j in range(self.w2.shape[1]):
        self.w2[i, j] = self.w2[i, j] - self.h_parameter
        cost = self.compute_cost(X, Y)
        self.w2[i, j] = self.w2[i, j] + 2 * self.h_parameter
        w2_grad[i, j] = (self.compute_cost(X, Y) - cost) / (2 * self.h_parameter)
        self.w2[i, j] = self.w2[i, j] - self.h_parameter
    return b1_grad, b2_grad, w1_grad, w2_grad

    def cyclic_learning_rate(self, t):
      if t < (2 * int(t / (2 * self.ns)) + 1) * self.ns:
        return self.learning_rate_min + (t - 2 * int(t / (2 * self.ns)) * self.ns) / self.ns * (self.learning_rate_max - self.learning_rate_min)
      else:
        return self.learning_rate_max - (t- (2 * int(t / (2 * self.ns)) + 1) * self.ns) / self.ns * (self.learning_rate_max - self.learning_rate_min)
