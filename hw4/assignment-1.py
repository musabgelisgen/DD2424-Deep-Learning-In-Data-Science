import numpy as np
import matplotlib.pyplot as plt
import pickle
from math import ceil
from assignment import NeuralNetwork

def main():
    book_data, book_characters, length = load_data()
    neuralNetwork = NeuralNetwork(length, book_characters)
    neuralNetwork.fit(book_data)
    #neuralNetwork.test_gradient(book_data[:net.seq_length], book_data[1:net.seq_length+1])

def load_data():
    book_data = ''
    with open("/Users/musabgelisgen/Desktop/goblet_book.txt") as fo:
        lines = fo.readlines()
    for line in lines:
        book_data = book_data + line

    book_characters = []

    for i in range(0, len(book_data)):
        if not(book_data[i] in book_characters):
            book_characters.append(book_data[i])

    length = len(book_characters)

    return book_data, book_characters, length

def plot():
    with open("./loss_final.npz.npy") as f:
        loss = list(np.load("loss_final.npz.npy").reshape(310072, 1))
    loss_plot = plt.plot(loss, label="Training loss")
    # print(loss[310070])
    plt.xlabel('Epochs')
    plt.xlim(0, 315000)
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()


    class NeuralNetwork:
        def __init__(self, K, char_list):
            # params
            self.m = 100
            self.eta = 0.1
            self.seq_length = 25
            self.d = K
            self.K = self.d
            self.eps = 1e-8
            self.number_of_epochs = 7
            self.char_list = char_list
            self.h_0 = np.zeros((self.m, 1))

            #network
            self.RNN_b = np.zeros((self.m, 1))
            self.RNN_b = np.zeros(self.RNN_b.shape)
            self.RNN_c = np.zeros((self.K, 1))
            self.RNN_c = np.zeros(self.RNN_c.shape)
            self.RNN_V = np.zeros((self.K, self.m))
            self.RNN_V = np.random.normal(0, 0.01, self.RNN_V.shape)
            self.RNN_U = np.zeros((self.m, self.K))
            self.RNN_U = np.random.normal(0, 0.01, self.RNN_U.shape)
            self.RNN_W = np.zeros((self.m, self.m))
            self.RNN_W = np.random.normal(0, 0.01, self.RNN_W.shape)

            self.grad_b = np.zeros((self.m, 1))
            self.grad_c = np.zeros((self.K, 1))
            self.grad_V = np.zeros((self.K, self.m))
            self.grad_U = np.zeros((self.m, self.K))
            self.grad_W = np.zeros((self.m, self.m))

            self.m_b = np.zeros((self.m, 1))
            self.m_c = np.zeros((self.K, 1))
            self.m_V = np.zeros((self.K, self.m))
            self.m_U = np.zeros((self.m, self.K))
            self.m_W = np.zeros((self.m, self.m))

        def pass_forward(self, x, h, b, c, W, U, V):
            ht = h
            H = np.zeros((self.m, x.shape[1]))
            P = np.zeros((self.K, x.shape[1]))
            A = np.zeros((self.m, x.shape[1]))
            for t in range(0, x.shape[1]):
                a = np.dot(W, ht) + np.dot(U, x[:, [t]]) + b
                ht = np.tanh(a)
                o = np.dot(V, ht) + c
                p = np.exp(o) / sum(np.exp(o))
                H[:, [t]] = ht
                P[:, [t]] = p
                A[:, [t]] = a
            return P, H, A

        def compute_cost(self, P, Y):
            loss_sum = 0
            for i in range(P.shape[1]):
                p = P[:, [i]]
                y = Y[:, [i]]
                loss_sum = loss_sum - np.log(np.dot(y.T, p))
            return loss_sum

        def test_gradient(self, X_chars, Y_chars):
            X = np.zeros((self.d, self.seq_length), dtype = int)
            Y = np.zeros((self.K, self.seq_length), dtype = int)

            for i in range(0, 2):
                self.h_0 = np.random.normal(0, 0.01, self.h_0.shape)

                for i in range(self.seq_length):
                    ind = np.zeros((self.char_list, 1), dtype=int)
                    ind[self.char_list.index(X_chars[i])] = 1
                    X[:, i] = ind.T

                    ind = np.zeros((self.char_list, 1), dtype=int)
                    ind[self.char_list.index(Y_chars[i])] = 1
                    Y[:, i] = ind.T

                P, H1, A = self.pass_forward(X, self.h_0, self.RNN_b, self.RNN_c, self.RNN_W, self.RNN_U, self.RNN_V)

                H_0 = np.zeros((self.m, self.seq_length))
                H_0[:, [0]] = self.h_0
                H_0[:, 1:] = H1[:, :-1]

                self.compute_gradients(P, X, Y, H1, H_0, A, self.RNN_V, self.RNN_W)
                grad_b, grad_c, grad_V, grad_U, grad_W = self.compute_gradients(X, Y, self.RNN_b, self.RNN_c, self.RNN_W, self.RNN_U, self.RNN_V)

                h = 1e-4
                # print(sum(abs(self.grad_b - grad_b)) / max(h, sum(abs(grad_b)) + sum(abs(self.grad_b))))
                # print(sum(abs(self.grad_c - grad_c)) / max(h, sum(abs(grad_c)) + sum(abs(self.grad_c))))
                # print(sum(sum(abs(self.grad_V - grad_V))) / max(h, sum(sum(abs(grad_V))) + sum(sum(abs(self.grad_V)))))
                # print(sum(sum(abs(self.grad_U - grad_U))) / max(h, sum(sum(abs(grad_U))) + sum(sum(abs(self.grad_U)))))
                # print(sum(sum(abs(self.grad_W - grad_W))) / max(h, sum(sum(abs(grad_W))) + sum(sum(abs(self.grad_W)))))

                self.m_b += np.multiply(self.grad_b, self.grad_b)
                self.m_c += np.multiply(self.grad_c, self.grad_c)
                self.m_U += np.multiply(self.grad_U, self.grad_U)
                self.m_W += np.multiply(self.grad_W, self.grad_W)
                self.m_V += np.multiply(self.grad_V, self.grad_V)

                self.RNN_b -= np.multiply(self.eta / np.sqrt(self.m_b + self.eps), self.grad_b)
                self.RNN_c -= np.multiply(self.eta / np.sqrt(self.m_c + self.eps), self.grad_c)
                self.RNN_U -= np.multiply(self.eta / np.sqrt(self.m_U + self.eps), self.grad_U)
                self.RNN_W -= np.multiply(self.eta / np.sqrt(self.m_W + self.eps), self.grad_W)
                self.RNN_V -= np.multiply(self.eta / np.sqrt(self.m_V + self.eps), self.grad_V)

                self.h_0 = H1[:,[-1]]

        def synthezise(self, x_0, h_0, n, b, c, W, U, V):
            Y = np.zeros((self.K, n))
            x = x_0
            h = h_0

            for i in range(0, n):
                p, h, _ = self.pass_forward(x, h, b, c, W, U, V)
                label = np.random.choice(self.K, p = p[:, 0])

                Y[label][i] = 1
                x = np.zeros(x.shape)
                x[label] = 1
            return Y

        def compute_gradients(self, P, X, Y, H, H_0, A, V, W):
            G = -(Y.T - P.T).T

            self.grad_V = np.dot(G, H.T)
            self.grad_c = np.sum(G, keepdims=True, axis=-1)

            a = np.zeros((X.shape[1], self.m))
            b = np.zeros((self.m, X.shape[1]))

            a[-1] = np.dot(G.T[-1], V)
            b[:,-1] = np.multiply(a[-1].T, (1 - np.multiply(np.tanh(A[:, -1]), np.tanh(A[:, -1]))))

            for r in range(X.shape[1] - 2, -1, -1):
                a[r] = np.dot(G.T[r], V) + np.dot(b[:, r + 1], W)
                b[:,r] = np.multiply(a[r].T, (1 - np.multiply(np.tanh(A[:, r]), np.tanh(A[:, r]))))

            self.grad_W = np.dot(b, H_0.T)
            self.grad_U = np.dot(b, X.T)
            self.grad_b = np.sum(b, axis = -1, keepdims = True)

            self.grad_b = np.where(self.grad_b < 5, self.grad_b, 5)
            self.grad_b = np.where(self.grad_b > -5, self.grad_b, -5)
            self.grad_c = np.where(self.grad_c < 5, self.grad_c, 5)
            self.grad_c = np.where(self.grad_c > -5, self.grad_c, -5)
            self.grad_V = np.where(self.grad_V < 5, self.grad_V, 5)
            self.grad_V = np.where(self.grad_V > -5, self.grad_V, -5)
            self.grad_U = np.where(self.grad_U < 5, self.grad_U, 5)
            self.grad_U = np.where(self.grad_U > -5, self.grad_U, -5)
            self.grad_W = np.where(self.grad_W < 5, self.grad_W, 5)
            self.grad_W = np.where(self.grad_W > -5, self.grad_W, -5)

        def fit(self, book_data):
            smooth_loss = 0
            iteration = 0
            losses = []
            length = len(book_data)
            nb_seq = ceil(float(length - 1) / float(self.seq_length))
            int synth = 200

            for i in range(0, self.number_of_epochs):
                e = 0
                hprev = np.random.normal(0, 0.01, self.h_0.shape)

                for j in range(0, int(nb_seq)):
                    if j != nb_seq-1:
                        X_chars = book_data[e:e + self.seq_length]
                        Y_chars = book_data[e + 1:e + self.seq_length + 1]
                        e = e + self.seq_length
                    else:
                        X_chars = book_data[e:length - 2]
                        Y_chars = book_data[e + 1:length - 1]
                        e = length

                    X = np.zeros((self.d, len(X_chars)), dtype = int)
                    Y = np.zeros((self.K, len(X_chars)), dtype = int)
                    length_2 = len(X_chars)

                    for i in range(0, length_2):
                        ind = np.zeros((len(self.char_list), 1), dtype = int)
                        ind[self.char_list.index(X_chars[i])] = 1
                        X[:, i] = ind.T

                        ind = np.zeros((len(self.char_list), 1), dtype = int)
                        ind[self.char_list.index(Y_chars[i])] = 1
                        Y[:, i] = ind.T

                    P, H1, A = self.pass_forward(X, hprev, self.RNN_b, self.RNN_c, self.RNN_W, self.RNN_U, self.RNN_V)

                    H_0 = np.zeros((self.m, len(X_chars)))
                    H_0[:, [0]] = self.h_0
                    H_0[:, 1:] = H1[:, :-1]

                    self.compute_gradients(P, X, Y, H1, H_0, A, self.RNN_V, self.RNN_W)

                    loss = self.compute_cost(P, Y)
                    if smooth_loss == 0:
                        smooth_loss = loss
                    else:
                        smooth_loss = 0.999 * smooth_loss + 0.001 * loss

                    self.m_b += np.multiply(self.grad_b, self.grad_b)
                    self.m_c += np.multiply(self.grad_c, self.grad_c)
                    self.m_U += np.multiply(self.grad_U, self.grad_U)
                    self.m_V += np.multiply(self.grad_V, self.grad_V)
                    self.m_W += np.multiply(self.grad_W, self.grad_W)

                    self.RNN_b -= np.multiply(self.eta / np.sqrt(self.m_b + self.eps), self.grad_b)
                    self.RNN_c -= np.multiply(self.eta / np.sqrt(self.m_c + self.eps), self.grad_c)
                    self.RNN_V -= np.multiply(self.eta / np.sqrt(self.m_V + self.eps), self.grad_V)
                    self.RNN_U -= np.multiply(self.eta / np.sqrt(self.m_U + self.eps), self.grad_U)
                    self.RNN_W -= np.multiply(self.eta / np.sqrt(self.m_W + self.eps), self.grad_W)

                    hprev = H1[:, [-1]]

                    if iteration % 1000 == 0:
                        print(iteration)

                    losses.append(smooth_loss)
                    if iteration % 10000 == 0:
                        Y_temp = self.synthezise(X[:, [0]], hprev, synth, self.RNN_b, self.RNN_c, self.RNN_W, self.RNN_U, self.RNN_V)
                        string = ""
                        for i in range(Y_temp.shape[1]):
                            string = string + self.char_list[np.argmax(Y_temp[:, [i]])]
                        print(string)
                    iteration = iteration + 1

            np.save("loss_final.npz", losses)

            ind = np.zeros((len(self.char_list), 1), dtype=int)
            ind[self.char_list.index("A")] = 1

            synth = 1000
            Y_temp = self.synthezise(ind, self.h_0, synth, self.RNN_b, self.RNN_c, self.RNN_W, self.RNN_U, self.RNN_V)
            string = ""
            for i in range(0, Y_temp.shape[1]):
                string = string + self.char_list[np.argmax(Y_temp[:, [i]])]
            print(string)

        def compute_gradients(self, X, Y, b, c, W, U, V):
            grad_b = np.zeros((self.m, 1))
            grad_c = np.zeros((self.K, 1))
            grad_V = np.zeros((self.K, self.m))
            grad_U = np.zeros((self.m, self.K))
            grad_W = np.zeros((self.m, self.m))

            h = 1e-4

            for i in range(0, b.shape[0]):
                b_temp = np.copy(b)
                b_temp[i] = b_temp[i] - h
                P, _, _, = self.pass_forward(X, self.h_0, b_temp, c, W, U, V)
                c1 = self.compute_cost(P, Y)
                b_temp = np.copy(b)
                b_temp[i] = b_temp[i] + h
                P, _, _, = self.pass_forward(X, self.h_0, b_temp, c, W, U, V)
                grad_b[i] = (self.compute_cost(P, Y) - c1) / (2 * h)

            for i in range(0, c.shape[0]):
                c_temp = np.copy(c)
                c_temp[i] = c_temp[i] - h
                P, _, _, = self.pass_forward(X, self.h_0, b, c_temp, W, U, V)
                c1 = self.compute_cost(P, Y)
                c_temp = np.copy(c)
                c_temp[i] = c_temp[i] + h
                P, _, _, = self.pass_forward(X, self.h_0, b, c_temp, W, U, V)
                grad_c[i] = (self.compute_cost(P, Y) - c1) / (2 * h)

            for i in range(0, V.shape[0]):
                for j in range(0, V.shape[1]):
                    V_temp = np.copy(V)
                    V_temp[i][j] = V_temp[i][j] - h
                    P, _, _, = self.pass_forward(X, self.h_0, b, c, W, U, V_temp)
                    c1 = self.compute_cost(P, Y)
                    V_temp = np.copy(V)
                    V_temp[i][j] = V_temp[i][j] + h
                    P, _, _, = self.pass_forward(X, self.h_0, b, c, W, U, V_temp)
                    grad_V[i][j] = (self.compute_cost(P, Y) - c1) / (2 * h)

            for i in range(0, U.shape[0]):
                for j in range(0, U.shape[1]):
                    U_temp = np.copy(U)
                    U_temp[i][j] = U_temp[i][j] - h
                    P, _, _, = self.pass_forward(X, self.h_0, b, c, W, U_temp, V)
                    c1 = self.compute_cost(P, Y)
                    U_temp = np.copy(U)
                    U_temp[i][j] = U_temp[i][j] + h
                    P, _, _, = self.pass_forward(X, self.h_0, b, c, W, U_temp, V)
                    grad_U[i][j] = (self.compute_cost(P, Y) - c1) / (2 * h)

            for i in range(0, W.shape[0]):
                for j in range(0, W.shape[1]):
                    W_temp = np.copy(W)
                    W_temp[i][j] = W_temp[i][j] - h
                    P, _, _, = self.pass_forward(X, self.h_0, b, c, W_temp, U, V)
                    c1 = self.compute_cost(P, Y)
                    W_temp = np.copy(W)
                    W_temp[i][j] = W_temp[i][j] + h
                    P, _, _, = self.pass_forward(X, self.h_0, b, c, W_temp, U, V)
                    grad_W[i][j] = (self.compute_cost(P, Y) - c1) / (2 * h)

            return grad_b, grad_c, grad_V, grad_U, grad_W
