from NN import NN
import random
import matplotlib.pyplot as plt
import numpy as np


def main():
    choice = ""

    while choice != "q":
        choice = input("Type 1 for weights initialization comparaison, 2 for the parameters search,"
                       "3 for the finite gradient, 4 for the test results on the best "
                       "hyper-parameters and q to quit.")

        if choice == str(1):
            compare_weight_initialization_methods()
        if choice == str(2):
            random_search(10)
        if choice == str(3):
            plot_finite_gradient()
        if choice == str(4):
            test_best_model()

    return None


def compare_weight_initialization_methods():
    loss_init = []

    mlp = NN(784, 10, data_path='mnist.pkl.npy', hidden_dims=(600, 600), activation_type='relu')

    loss_init.append(mlp.train(10, 'zeros'))
    mlp.reset()
    loss_init.append(mlp.train(10, 'normal_dist'))
    mlp.reset()
    loss_init.append(mlp.train(10,'glorot'))

    plt.plot(loss_init[0], label='Zeros')
    plt.plot(loss_init[1], label='Dist. Normale')
    plt.plot(loss_init[2], label='Glorot')
    plt.xlabel("epochs")
    plt.ylabel("training error")
    plt.legend(loc='upper right')

    plt.show()


def random_search(num_search):
    activations = ['relu', 'sigmoid', 'tanh']

    for i in range(num_search):

        # Random initialization of hyper-parameters
        dim_l1 = random.randint(450, 650)
        dim_l2 = random.randint(450, 650)
        activation = activations[random.randint(0, 2)]
        learning_rate = random.uniform(0.005, 0.02)

        batch_random = random.randint(4, 6)
        mini_batch_size = 2**batch_random

        # Creation and training of the NN
        mlp = NN(784, 10, data_path='mnist.pkl.npy', hidden_dims=(dim_l1, dim_l2), activation_type=activation,
                 batch_size=mini_batch_size, learning_rate=learning_rate)

        print("Dim:("+ str(dim_l1) + "," + str(dim_l2) + "), activation: " + activation + " , learning rate: "
              + str(learning_rate) + ", batch_size = " + str(mini_batch_size))

        mlp.train(10, 'glorot')


def plot_finite_gradient():

    mlp = NN(784, 10, data_path='mnist.pkl.npy', hidden_dims=(600, 600), activation_type='relu')
    mlp.train(10, 'glorot')

    N = np.array([1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500,
                 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000])
    mlp.plot_finite_gradient(N)


def test_best_model(data_path='mnist.pkl.npy'):
    mlp = NN(784, 10, data_path='mnist.pkl.npy', hidden_dims=(544, 570), activation_type='relu',
             learning_rate=0.01213, batch_size=64)
    mlp.train(10,'glorot')

    data = np.load(data_path)[2]

    mlp.test(data)

if __name__ == '__main__':
    main()
