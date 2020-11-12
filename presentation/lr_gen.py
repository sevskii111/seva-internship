import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
my_path = pathlib.Path(__file__).parent.absolute()


def f(x):
    return 2 * x + 3

def f_with_noise(func, x):
    return func(x) + np.random.uniform(-1, 1, x.shape) * np.random.uniform(0, 5, x.shape)

def line(x, k, b):
    return x * k + b

def MSE(Y, Y_real):
    return np.mean(np.square(Y - Y_real))

X = np.linspace(0, 20)
Y = f_with_noise(f, X)
Y_real = f(X)

lr = 0.0005

k = 0.001 * np.random.randn()
b = 0.001 * np.random.randn()

k_history = []
b_history = []

for i in range(10):
    Y_pred = line(X, k, b)
    k_history.append(k)
    b_history.append(b)

    loss = MSE(Y_real, Y_pred)

    k_grad = -2 / len(X) * np.sum((Y * (Y - Y_pred)))
    b_grad = -2 / len(X) * np.sum(Y - Y_pred)

    plt.xlim(0, 20)
    plt.ylim(-1, 50)
    plt.plot(X, Y, 'bo')
    plt.plot(X, Y_real, 'g', linewidth=1.0, label='y = 2x + 3')
    plt.plot(X, Y_pred, 'r', linewidth=2.0, label='y_predicted = {:.2f}x + {:.2f}'.format(k, b))
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title('epoch={}, MSE={:.2f}, ∇=({:.2f}, {:.2f})'.format(i + 1, loss, k_grad, b_grad))
    plt.legend()
    plt.savefig(os.path.join(my_path, 'img/lin_{}.png'.format(i)))
    plt.clf()

    k -= k_grad * lr
    b -= b_grad * lr


for i in range(100000):
    Y_pred = line(X, k, b)
    k_grad = -2 / len(X) * np.sum((Y * (Y - Y_pred)))
    b_grad = -2 / len(X) * np.sum(Y - Y_pred)

    k -= k_grad * lr
    b -= b_grad * lr

Y_pred = line(X, k, b)
k_history.append(k)
b_history.append(b)
loss = MSE(Y_real, Y_pred)
plt.xlim(0, 20)
plt.ylim(-1, 50)
plt.plot(X, Y, 'bo')
plt.plot(X, Y_real, 'g', linewidth=1.0, label='y = 2x + 3')
plt.plot(X, Y_pred, 'r', linewidth=2.0, label='y_predicted = {:.2f}x + {:.2f}'.format(k, b))
plt.ylabel('y')
plt.xlabel('x')
plt.title('epoch={}, MSE={:.2f}, ∇=({:.2f}, {:.2f})'.format(100010, loss, k_grad, b_grad))
plt.legend()
plt.savefig(os.path.join(my_path, 'img/lin_{}.png'.format(11)))
plt.clf()

k_history_file = open(os.path.join(my_path, "k_history.txt"), "w")
k_history_file.write(','.join(map(str, k_history)))
k_history_file.close()

b_history_file = open(os.path.join(my_path, "b_history.txt"), "w")
b_history_file.write(','.join(map(str, b_history)))
b_history_file.close()