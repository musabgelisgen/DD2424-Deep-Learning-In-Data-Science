
def main():
  np.random.seed(20)

  X_train, y_train, Y_train, X_validation, y_validation, Y_validation = get_train_val_data()
  X_test, y_test, Y_test = load_data(batch_file = "cifar-10-batches-py/test_batch", number=None, number_f=None, k=10)
  neural_network = NeuralNetwork(X_train, Y_train)
  neural_network.fit(X_train, Y_train, X_validation, Y_validation)
  Y_pred_test,_,_,_,_,_,_ = neural_network.evaluate(X_test)
  accuracy = neural_network.compute_accuracy(Y_pred_test, Y_test)
  print(accuracy)

def main_lambdasearch():
  np.random.seed(20)

  X_train, y_train , Y_train = load_data(batch_file = "cifar-10-batches-py/data_batch_1", number=None, number_f=None, k=10)
  X_test, y_test, Y_test = load_data(batch_file = "cifar-10-batches-py/test_batch", number=None, number_f=None, k=10)
  X_validation, y_validation , Y_validation = load_data(batch_file = "cifar-10-batches-py/data_batch_2", number=None, number_f=None, k=10)

  lamda_fine = [5*1e-3, 6*1e-3, 7*1e-3, 8*1e-3, 9*1e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2]

  for lamda in lamda_fine:
    params = {
      "lamda": lamda,
      "epochs": 40
    }
    neural_network = NeuralNetwork(X_train, Y_train, **params)
    neural_network.fit(X_train, Y_train, X_validation, Y_validation)
    Y_pred_val,_,_,_,_,_,_ = neural_network.evaluate(X_validation)
    val_acc = neural_network.compute_accuracy(Y_pred_val, Y_validation)
    print(lamda, ": ", val_acc)

def get_train_val_data():
  X1, y1, Y1 = load_data(batch_file="cifar-10-batches-py/data_batch_1", number=None, number_f=None, k=10)
  X2, y2, Y2 = load_data(batch_file="cifar-10-batches-py/data_batch_2", number=None, number_f=None, k=10)
  X3, y3, Y3 = load_data(batch_file="cifar-10-batches-py/data_batch_3", number=None, number_f=None, k=10)
  X4, y4, Y4 = load_data(batch_file="cifar-10-batches-py/data_batch_4", number=None, number_f=None, k=10)
  X5, y5, Y5 = load_data(batch_file="cifar-10-batches-py/data_batch_5", number=None, number_f=None, k=10)

  X_train = np.concatenate((X1, X2, X3, X4, X5), axis=1)[:, :-500]
  y_train = np.concatenate((y1, y2, y3, y4, y5))[:-500]
  Y_train = np.concatenate((Y1, Y2, Y3, Y4, Y5), axis=1)[:, :-500]
  X_validation = np.concatenate((X1, X2, X3, X4, X5), axis=1)[:, -500:]
  y_validation = np.concatenate((y1, y2, y3, y4, y5))[-500:]
  Y_validation = np.concatenate((Y1, Y2, Y3, Y4, Y5), axis=1)[:, -500:]
  return X_train, y_train, Y_train, X_validation, y_validation, Y_validation


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


    import numpy as np
    import matplotlib.pyplot as plt

class NeuralNetwork:

      def __init__(self, data, targets, **kwargs):
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
            "hidden_layer_sizes": [50, 30, 20, 20, 10, 10, 10, 10],
            # "hidden_layer_sizes":[50, 50],
            "alpha": 0.7,
            "learning_rate_min": 1e-5,
            "learning_rate_max": 1e-1
        }

        for var, default in variables.items():
            setattr(self, var, kwargs.get(var, default))

        self.batch_norm_flag = True
        # self.batch_norm_flag = False
        self.sigma_weights == "square_root_m"
        # self.batch_norm_flag = False

        # vector
        self.d = data.shape[0]
        self.n = data.shape[1]
        self.k = targets.shape[0]
        self.m = self.hidden_layer_sizes
        self.number_of_h_layers = len(self.hidden_layer_sizes)

        weight_numbers = self.number_of_h_layers+1
        sigma_weights_for_w = np.zeros(weight_numbers)
        if type(self.sigma_weights == 'float'):
          for i in range(weight_numbers):
            sigma_weights_for_w[i] = self.sigma_weights
        if self.sigma_weights == "square_root_m":
          sigma_weights_for_w[0] = 1 / np.sqrt(self.d)
          for i in range(1, weight_numbers):
            sigma_weights_for_w[i] = 1 / np.sqrt(self.m[i-1])
        w = [np.zeros((2, 2))] * weight_numbers
        w[0] = np.random.normal(self.m_weights, sigma_weights_for_w[0], (self.m[0], self.d))
        for i in range(1, weight_numbers-1):
          w[i] = np.random.normal(self.m_weights, sigma_weights_for_w[i], (self.m[i], self.m[i - 1]))
        w[weight_numbers-1] = np.random.normal(self.m_weights, sigma_weights_for_w[weight_numbers - 1], (self.k, self.m[weight_numbers - 2]))
        self.w = w

        layer_n = self.number_of_h_layers
        b = [[0] * (layer_n+1)]
        for i in range(layer_n):
          b[i] = np.zeros((self.m[i], 1))
        b[layer_n] = np.zeros((self.k, 1))
        self.b = b

        self.ns = 5 * int(self.n / self.batch_size)

        weight_numbers = self.number_of_h_layers+1
        sigma_weights_for_w = np.zeros(weight_numbers)
        if self.sigma_weights == "square_root_m":
          sigma_weights_for_w[0] = 1 / np.sqrt(self.d)
          for i in range(1, weight_numbers):
            sigma_weights_for_w[i] = 1/np.sqrt(self.m[i-1])
        elif type(self.sigma_weights == 'float'):
          for i in range(weight_numbers):
            sigma_weights_for_w[i] = self.sigma_weights
        w = [np.zeros((2, 2))] * weight_numbers
        w[0] = np.random.normal(self.m_weights, sigma_weights_for_w[0], (self.m[0], self.d))

        rng = weight_numbers-1
        for i in range(1, rng):
          w[i] = np.random.normal(self.m_weights, sigma_weights_for_w[i], (self.m[i], self.m[i - 1]))
        w[weight_numbers - 1] = np.random.normal(self.m_weights, sigma_weights_for_w[weight_numbers - 1], (self.k, self.m[weight_numbers-2]))
        self.beta = w

        gamma = [0] * (self.number_of_h_layers)
        for i in range(0, self.number_of_h_layers):
          gamma[i] = np.random.normal(self.m_weights, np.sqrt(2 / self.hidden_layer_sizes[i]), (self.hidden_layer_sizes[i], 1))
        self.gamma = gamma

        self.average_mu = None
        self.average_var = None

      def fit(self, X_train, Y_train, X_validation, Y_validation):

        self.cost_hist_training = []
        self.cost_hist_validation = []
        self.acc_hist_training = []
        self.acc_hist_validation = []

        num_batches = int(self.n / self.batch_size)
        for i in range(self.epochs):
          for j in range(num_batches):
            j_start = j * self.batch_size
            j_end = j * self.batch_size + self.batch_size
            X_batch = X_train[:, j_start:j_end]
            Y_batch = Y_train[:, j_start:j_end]
            Y_pred, actual_h, X_batch, s, normalization_s, mu, var = self.evaluate(X_batch)
            grad_b, grad_w, grad_beta, grad_gamma = self.compute_gradients(X_batch, Y_batch, Y_pred, actual_h, s, normalization_s, mu, var)

            for k in range(self.number_of_h_layers+1):
              self.w[k] -= self.lr * grad_w[k]
              self.b[k] -= self.lr * grad_b[k]

            if self.batch_norm_flag:
              for k in range(0, self.number_of_h_layers):
                self.beta[k] -= self.lr * grad_beta[k]
                self.gamma[k] -= self.lr * grad_gamma[k]

            self.lr = self.cyclic_lr(i * num_batches + j)
          self.report_perf(i, X_train, Y_train, X_validation, Y_validation)
        self.plot_cost_and_acc()

      def check_gradients(self, X, Y):
        grad_w_num = np.zeros((self.k, self.d))
        Y_pred, h_act, _, s, normalization_s, mu, var = self.evaluate(X)
        grad_b, grad_w, grad_beta, grad_gamma  = self.compute_gradients(X, Y, Y_pred, h_act, s, normalization_s, mu, var)

        grad_b_num, grad_w_num = self.compute_gradient_numerically(X, Y)
        # grad_b_num, grad_w_num= self.compute_gradient_num_slow(X, Y)

        for k in range(self.number_of_h_layers + 1):
            grad_vec = grad_w[k].flatten()
            grad_num_vec = grad_w_num[k].flatten()
            x_w = np.arange(1, grad_vec.shape[0] + 1)
            plt.bar(x_w, grad_vec, 0.35, label='Analytical gradient')
            plt.bar(x_w + 0.35, grad_num_vec, 0.35, label=centered_diff)
            plt.legend()
            plt.title("Gradient check")
            plt.show()
            rel_error = abs((grad_vec+self.h_parameter**2) / (grad_num_vec+self.h_parameter**2) - 1)

            grad_vec = grad_b[k].flatten()
            grad_num_vec = grad_num[k].flatten()
            x_b = np.arange(1, grad_b[k].shape[0] + 1)
            plt.bar(x_b, grad_vec, 0.35, label='Analytical gradient')
            plt.bar(x_b + 0.35, grad_num_vec, 0.35, label=centered_diff)
            plt.legend()
            plt.title("Gradient check")
            plt.show()
            rel_error = abs((grad_vec+self.h_parameter**2) / (grad_num_vec+self.h_parameter**2) - 1)

            print(np.mean(rel_error))

      def evaluate(self, X):
        normalization_s = [0] * (self.number_of_h_layers)
        var = [0] * (self.number_of_h_layers)
        mu = [0] * (self.number_of_h_layers)
        actual_h = [0] * (self.number_of_h_layers)
        s = [0] * (self.number_of_h_layers + 1)
        final_s = [0] * (self.number_of_h_layers)

        s[0] = np.dot(self.w[0], X) + self.b[0]
        if self.batch_norm_flag:
            var[0] = 1/ s[0].shape[1] * np.sum(((s[0].T - mu[0]).T)**2, axis = 1)
            mu[0] = 1 / s[0].shape[1] * np.sum(s[0], axis = 1)
            normalization_s[0] = self.batch_norm(s[0], mu[0], var[0])
            final_s[0] = self.gamma[0] * normalization_s[0] + self.beta[0]

        if self.batch_norm_flag == False:
            actual_h[0] = np.maximum(0, s[0])
        else:
            actual_h[0] = np.maximum(0, final_s[0])

        for i in range(1, self.number_of_h_layers):
          s[i] = np.dot(self.w[i], actual_h[i-1]) + self.b[i]
          if self.batch_norm_flag == False:
            actual_h[i] = np.maximum(0, s[i])
          else:
            var[i] = 1 / s[i].shape[1] * np.sum(((s[i].T - mu[i]).T)**2, axis = 1)
            mu[i] = 1 / s[i].shape[1] * np.sum(s[i], axis = 1)
            normalization_s[i] = self.batch_norm(s[i], mu[i], var[i])
            final_s[i] = self.gamma[i] * normalization_s[i] + self.beta[i]
            actual_h[i] = np.maximum(0, final_s[i])

        final_layer = self.number_of_h_layers
        s[final_layer] = np.dot(self.w[final_layer], actual_h[final_layer-1]) + self.b[final_layer]

        if self.batch_norm_flag:
          if self.average_mu != None:
              self.average_mu = []
              for l in range(len(mu)):
                self.average_mu.append(self.alpha * self.average_mu[l] + (1 - self.alpha) * mu[l])
          else:
            self.average_mu = mu
          if self.average_var != None:
              self.average_var = []
              for l in range(len(var)):
                self.average_var.append(self.alpha * self.average_var[l] + (1 - self.alpha) * var[l])
          else:
            self.average_var = var

        return self.softmax(s[final_layer]), actual_h, X, s, normalization_s, mu, var

      def softmax(self, Y_predict):
        return np.exp(Y_predict) / np.dot(np.ones(Y_predict.shape[0]).T, np.exp(Y_predict))

      def batch_norm(self, s, mu, var):
    # np.dot(np.diag(np.power((var + self.h_parameter),(-1/2))), (s.T - mu).T)
        return np.dot(np.diag(np.power((var + self.h_parameter),(-1/2))), (s.T - mu).T)

      def compute_cost(self, X, Y):
        Y_pred, actual_h,_,_,_,_,_ = self.evaluate(X)
        regularization = 0
        for i in range(0, self.number_of_h_layers + 1):
          regularization += self.lamda * np.sum(self.w[i]**2)
        return np.sum(-np.log(np.sum(Y * Y_pred, axis=0)), axis=0) / X.shape[1] + regularization

      def cross_entropy(self, Y, Y_pred):
        return np.sum(-np.log(np.sum(Y * Y_pred, axis=0)), axis=0)

      def compute_accuracy(self, Y_predict, Y_actual):
        correct = len(np.where(np.array(np.argmax(Y_actual, axis=0)) == np.array(np.argmax(Y_predict, axis=0)))[0])
        return correct / np.array(np.argmax(Y_actual, axis=0)).shape[0]

      def compute_gradients(self, X_batch, y_true_batch, y_pred_batch, actual_h):
        grad_b = [0] * (self.number_of_h_layers+1)
        grad_w = [0] * (self.number_of_h_layers+1)
        grad_gamma = [0] * (self.number_of_h_layers)
        grad_beta = [0] * (self.number_of_h_layers)

        grad_w[-1] = (1 / self.batch_size) * np.dot(y_pred_batch-y_true_batch, actual_h[self.number_of_h_layers-1].T) + 2 * self.lamda*self.w[self.number_of_h_layers]
        grad_b[-1] = (1 / self.batch_size) * np.dot(y_pred_batch-y_true_batch, np.ones((self.batch_size,1)))

        grad_batch = np.dot(self.w[-1].T, y_pred_batch-y_true_batch)

        layers_input = actual_h[self.number_of_h_layers-1]
        h_act_ind = np.zeros(layers_input.shape)
        for k in range(0, layers_input.shape[0]):
          for j in range(0, layers_input.shape[1]):
            if layers_input[k,j] > 0:
              h_act_ind[k, j] = 1
        grad_batch = grad_batch * h_act_ind

        for l in reversed(range(self.number_of_h_layers)):
          if self.batch_norm_flag:
            grad_gamma[l] = (1 / self.batch_size) * np.dot((grad_batch * normalization_s[l]), np.ones(self.batch_size)).reshape(-1,1)
            grad_beta[l] = (1 / self.batch_size) * np.dot(grad_batch, np.ones(self.batch_size)).reshape(-1,1)

            grad_batch = grad_batch * np.dot(self.gamma[l], np.ones((self.batch_size,1)).T)
            grad_batch = self.batch_norm_helper(grad_batch, s[l], mu[l], var[l])

          if l != 0:
            grad_w[l] = (1 / self.batch_size) * np.dot(grad_batch, actual_h[l-1].T) + 2 * self.lamda*self.w[l]
          else:
            grad_w[l] = (1 / self.batch_size) * np.dot(grad_batch, X_batch.T) + 2 * self.lamda*self.w[l]

          grad_b[l] = (1 / self.batch_size) * np.dot(grad_batch, np.ones((self.batch_size,1)))

          if l > 0:
            h_act_ind = np.zeros(actual_h[l-1].shape)
            for k in range(actual_h[l-1].shape[0]):
              for j in range(actual_h[l-1].shape[1]):
                if actual_h[l-1][k,j] > 0:
                  h_act_ind[k, j] = 1
            grad_batch = np.dot(self.w[l].T, grad_batch) * h_act_ind

        return grad_b, grad_w, grad_beta, grad_gamma

      def batch_norm_helper(self, grad_batch, s_batch, mu, var):
        c = np.dot((grad_batch * np.outer(((var + self.h_parameter)**(-1.5)).T, np.ones((self.batch_size,1))) * s_batch - np.outer(mu, np.ones((self.batch_size,1)))), np.ones((self.batch_size,1)))
        grad_batch = grad_batch * np.outer(((var + self.h_parameter)**(-0.5)).T, np.ones((self.batch_size,1))) - 1 / self.batch_size * np.dot(bigG1, np.ones((self.batch_size,1)))
        return grad_batch - 1 / self.batch_size * (s_batch - np.outer(mu, np.ones((self.batch_size,1))) * np.outer(c, np.ones((self.batch_size))))

      def cyclic_learning_rate(self, t):
        if t < (2 * int(t / (2 * self.ns)) + 1) * self.ns:
          return self.learning_rate_min + (t - 2 * int(t / (2 * self.ns)) * self.ns) / self.ns * (self.learning_rate_max - self.learning_rate_min)
        else:
          return self.learning_rate_max - (t- (2 * int(t / (2 * self.ns)) + 1) * self.ns) / self.ns * (self.learning_rate_max - self.learning_rate_min)

      def report_performance(self, X_train, Y_train, X_validation, Y_validation, epoch):
        Y_pred_train, _, _, _, _, _, _ = self.evaluate(X_train)
        Y_pred_validation, _, _, _, _, _, _ = self.evaluate(X_validation)
        cost_train = self.compute_cost(X_train, Y_pred_train)
        acc_train = self.compute_accuracy(Y_pred_train, Y_train)
        cost_val = self.compute_cost(X_validation, Y_pred_validation)
        acc_val = self.compute_accuracy(Y_pred_val, Y_validation)

        self.cost_hist_training.append(cost_train)
        self.acc_hist_training.append(acc_train)
        self.cost_hist_validation.append(cost_val)
        self.acc_hist_validation.append(acc_val)
        print("Epochs: ", epoch, " Accuracy: ", acc_train, " Cost: ", cost_train)

      def plot_cost(self):
        leng = len(self.cost_hist_training) + 1
        x = list(range(1, leng))
        plt.plot(x, self.cost_hist_training, label = "Training loss")
        plt.plot(x, self.cost_hist_validation, label = "Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.ylim(0, 3)
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
        plt.ylim(0,0.8)
        plt.legend()
        plt.show()

      def plot_weights(self):
        for i in range(self.k):
          w_image = self.w1[i, :].reshape((32, 32, 3), order='F')
          plt.imshow(np.rot90(((w_image - w_image.min()) / (w_image.max() - w_image.min())), 3))
          plt.title(self.cifar10_labels[i])
          plt.show()

      def compute_gradient_numerically(self, X, Y):
        grad_w = [0] * (self.number_of_h_layers + 1)
        grad_b = [0] * (self.number_of_h_layers + 1)

        for k in range(0, self.number_of_h_layers):
            grad_b[k] = np.zeros((self.m[k], 1))
        grad_b[self.number_of_h_layers] = np.zeros((self.k, 1))

        grad_w[0] = np.zeros((self.m[0], self.d))
        for k in range(1, self.number_of_h_layers):
          grad_w[k] = np.zeros((self.m[k], self.m[k - 1]))
        grad_w[self.number_of_h_layers] = np.zeros((self.k, self.m[self.number_of_h_layers - 1]))

        c = self.compute_cost(X, Y)

        for k in range(self.number_of_h_layers + 1):
            for i in range(self.b[k].shape[0]):
              self.b[k][i] += self.h_parameter
              grad_b[k][i] = (self.compute_cost(X, Y) - c) / self.h_parameter
              self.b[k][i] -= self.h_parameter

            for i in range(self.w[k].shape[0]):
              for j in range(self.w[k].shape[1]):
                self.w[k][i,j] += self.h_parameter
                grad_w[k][i,j] = (self.compute_cost(X, Y) - c) / self.h_parameter
                self.w[k][i][j] -= self.h_parameter

        return grad_b, grad_w

      def compute_gradient_anatically(self, X, Y):
        grad_w = [0] * (self.number_of_h_layers + 1)
        grad_b = [0] * (self.number_of_h_layers + 1)

        for k in range(self.number_of_h_layers):
            grad_b[k] = np.zeros((self.m[k], 1))
        grad_b[self.number_of_h_layers] = np.zeros((self.k, 1))

        grad_w[0] = np.zeros((self.m[0], self.d))
        for k in range(1,self.number_of_h_layers):
            grad_w[k] = np.zeros((self.m[k], self.m[k - 1]))
        grad_w[self.number_of_h_layers] = np.zeros((self.k, self.m[self.number_of_h_layers-1]))

        for k in range(self.number_of_h_layers+1):
            for i in range(self.b[k].shape[0]):
              self.b[k][i] -= self.h_parameter
              c1 = self.compute_cost(X, Y)
              self.b[k][i] += 2 * self.h_parameter
              grad_b[k][i] = (self.compute_cost(X, Y) - c1) / (2 * self.h_parameter)
              self.b[k][i] -= self.h_parameter

            for i in range(self.w[k].shape[0]):
              for j in range(self.w[k].shape[1]):
                self.w[k][i, j] -= self.h_parameter
                c1 = self.compute_cost(X, Y)
                self.w[k][i, j] += 2 * self.h_parameter
                grad_w[k][i, j] = (self.compute_cost(X, Y) - c1) / (2 * self.h_parameter)
                self.w[k][i, j] -= self.h_parameter
        return grad_b, grad_w
