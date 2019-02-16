import numpy as np
import copy
import matplotlib.pyplot as plt

class NN (object):

    def __init__(self, input_size, num_class, data_path, learning_rate=0.00001, hidden_dims=(600, 600), batch_size=32,
                 activation_type='relu'):

        # Network initialization parameters
        self.data_path = data_path
        self.hidden_dims = hidden_dims
        self.input_size = input_size
        self.num_class = num_class

        # Network hyper-parameters
        self.learning_rate = learning_rate
        self.num_hidden = len(hidden_dims)
        self.batch_size = batch_size
        self.activation_type = activation_type

        # Network parameters
        self.params = {
            'w': [],
            'b': []
        }

        # Cache for backpropagation
        self.cache = []

        # Utils for loss
        self.l_vector = []
        self.plt_vector = []

    def initialize_weights(self, input_size, num_class, init_type):

        # Zero initialization
        if init_type == 'zeros':
            self.params['w'].append(np.zeros((input_size, self.hidden_dims[0])))
            self.params['w'].append(np.zeros((self.hidden_dims[0], self.hidden_dims[1])))
            self.params['w'].append(np.zeros((self.hidden_dims[1], num_class)))
            self.params['b'].append(np.zeros((1, self.hidden_dims[0])))
            self.params['b'].append(np.zeros((1, self.hidden_dims[1])))
            self.params['b'].append(np.zeros((1, num_class)))

        # Normal distribution initialization(with biases 0)
        if init_type == 'normal_dist':
            self.params['w'].append(np.random.normal(0, 1, (input_size, self.hidden_dims[0])))
            self.params['w'].append(np.random.normal(0, 1, (self.hidden_dims[0], self.hidden_dims[1])))
            self.params['w'].append(np.random.normal(0, 1, (self.hidden_dims[1], num_class)))
            self.params['b'].append(np.random.normal(0, 1,(1, self.hidden_dims[0])))
            self.params['b'].append(np.random.normal(0,1,(1, self.hidden_dims[1])))
            self.params['b'].append(np.random.normal(0,1,(1, num_class)))

        # Glorot initialization
        if init_type == 'glorot':
            d1 = np.sqrt(6./(input_size + self.hidden_dims[0]))
            d2 = np.sqrt(6./(self.hidden_dims[0] + self.hidden_dims[1]))
            do = np.sqrt(6./(self.hidden_dims[1] + num_class))

            self.params['w'].append(np.random.uniform(-d1, d1, (input_size, self.hidden_dims[0])))
            self.params['w'].append(np.random.uniform(-d2, d2, (self.hidden_dims[0], self.hidden_dims[1])))
            self.params['w'].append(np.random.uniform(-do, do, (self.hidden_dims[1], num_class)))
            self.params['b'].append(np.zeros((1, self.hidden_dims[0])))
            self.params['b'].append(np.zeros((1, self.hidden_dims[1])))
            self.params['b'].append(np.zeros((1, num_class)))

    def forward(self, x):
        # First pre-activation
        h1 = x.dot(self.params['w'][0]) + self.params['b'][0]

        # Activation for first non-linearity
        a1 = self.activation(h1)

        # Second pre-activation
        h2 = np.dot(a1, self.params['w'][1]) + self.params['b'][1]

        # Activation for second non-linearity
        a2 = self.activation(h2)

        # Output pre-activation
        o = np.dot(a2, self.params['w'][2]) + self.params['b'][2]

        # Caching activations for backpropagation
        self.cache.append(a1)
        self.cache.append(a2)

        # Returns the softmax
        return self.softmax(o)

    def activation(self, x):
        if self.activation_type == 'relu':
            return np.nan_to_num(np.maximum(x, 0, x))

        if self.activation_type == 'sigmoid':
            return np.nan_to_num(1 / (np.exp(-x) + 1))

        if self.activation_type == 'tanh':
            return np.nan_to_num((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))

    def activation_derivative(self, x):

        if self.activation_type == 'relu':
            return 1. * (x > 0)

        if self.activation_type == 'sigmoid':
            return self.activation(x)*(1 - self.activation(x))

        if self.activation_type == 'tanh':
            return 1 - np.square(self.activation(x))

    def loss(self, pred_vector, correct_label, batch_size, loss_type='cross-entropy'):
        if loss_type == 'cross-entropy':
            # Prevents infinite loss if the correct class has 0 probability.
            pred_vector = np.clip(pred_vector, 0.00001,0.99999)
            return np.average(-np.log(pred_vector[range(batch_size), correct_label]))

    def softmax(self, x):
        # Stabilize the Softmax
        x -= np.max(x, axis=1, keepdims=True)

        # Computes the Softmax function
        exp_score = np.exp(x)
        return exp_score/np.sum(exp_score, axis=1, keepdims=True)

    def backward(self, input, correct_label, prediction, mini_batch_size):
        # Calculates the loss
        loss = self.loss(prediction,correct_label, mini_batch_size)
        self.l_vector.append(loss)

        # Computes the derivatives of the output layer ( np.multiply(y, np.multiply(1-y, targets-y)))
        do = prediction
        do[range(mini_batch_size), correct_label] -= 1
        dwo = np.dot(self.cache[1].T, do)
        bo = np.sum(do, axis=0)

        # Computes the derivatives of the second layer
        dh2 = np.dot(do, self.params['w'][2].T) * self.activation_derivative(self.cache[1])
        dw2 = np.dot(self.cache[0].T, dh2)
        b2 = np.sum(dh2, axis=0)

        # Computes the derivatives of the first layer
        dh1 = np.dot(dh2, self.params['w'][1].T) * self.activation_derivative(self.cache[0])
        dw1 = np.dot(input.T, dh1)
        b1 = np.sum(dh1, axis=0)

        return [dw1, dw2, dwo], [b1, b2, bo]

    def update(self, grads, gbias):

        # Update output layer
        self.params['w'][2] -= self.learning_rate * grads[2]
        self.params['b'][2] -= self.learning_rate * gbias[2]

        # Update last hidden layer
        self.params['w'][1] -= self.learning_rate * grads[1]
        self.params['b'][1] -= self.learning_rate * gbias[1]

        # Update first hidden layer
        self.params['w'][0] -= self.learning_rate * grads[0]
        self.params['b'][0] -= self.learning_rate * gbias[0]

    def train(self, epochs, weight_init_method):
        loss_history = []

        # Data extraction
        data = np.load(self.data_path)[0]
        validation = np.load(self.data_path)[1]

        # Weights initialization
        self.initialize_weights(self.input_size, self.num_class, weight_init_method)

        # Training
        for epoch in range(0, epochs):
            for i in range(0, len(data[0]), self.batch_size):
                current_batch_size = self.batch_size
                batch_end_index = i + self.batch_size

                # Case where the batch size would cause to overflow the data
                if batch_end_index > len(data[0]):
                    batch_end_index = len(data[0])
                    current_batch_size = len(data[0]) - i

                # Initialize current training batch
                x = data[0][range(i, batch_end_index)]
                t = data[1][range(i, batch_end_index)]

                # Foward pass
                prediction = self.forward(x)

                # Backpropagation
                grads, gbias = self.backward(x,t,prediction,current_batch_size)

                # Weights update
                self.update(grads, gbias)

                # Cache clearing
                self.cache = []

            # Shuffle the training data
            randomize = np.arange(len(data[0]))
            np.random.shuffle(randomize)
            data[0] = data[0][randomize]
            data[1] = data[1][randomize]

            # Record loss
            loss_history.append(np.average(self.l_vector))
            self.l_vector = []

        # Test on validation data
        self.test(validation)
        self.cache = []
        # Returns average loss for each epoch
        return loss_history

    def test(self, test_set):
        results = 0
        for i in range(0, len(test_set[0]), self.batch_size):
            batch_end_index = i + self.batch_size

            # Case where the batch size would cause to overflow the data
            if batch_end_index > len(test_set[0]):
                batch_end_index = len(test_set[0])
                current_batch_size = len(test_set[0]) - i

            # Initialize current training batch
            x = test_set[0][range(i, batch_end_index)]

            # Foward pass
            prediction = self.forward(x)

            # Summing correct results
            results += np.sum(np.equal(np.argmax(prediction, axis=1), test_set[1][i:batch_end_index]).astype(int))

        print("Test results :" + str((results / len(test_set[0])) * 100) + "%")


    def plot_finite_gradient(self, N):

        # Loading the sample to use for evaluation
        sample = np.load(self.data_path)[2]
        x = sample[0][0]
        t = sample[1][0]
        x.shape = (1, len(x))

        # Setup variables for the calculations
        epsilon = 1/N
        last_layer_weights = self.params['w'][2]
        finite_gradient = np.zeros((len(epsilon), self.num_class))

        # Calculate true gradient
        prediction = self.forward(x)
        true_gradient, _ = self.backward(x, t, prediction, 1)

        for i in range(0, 10):
            for j in range(0, len(epsilon)):

                # Calculate L with -epsilon
                min_loss_weights = copy.deepcopy(last_layer_weights)
                min_loss_weights[:, i] -= epsilon[j]
                self.params['w'][2] = min_loss_weights

                prediction = self.forward(x)
                min_loss = self.loss(prediction, t, 1)

                # Calculate L with +epsilon
                max_loss_weights = copy.deepcopy(last_layer_weights)
                max_loss_weights[:, i] += epsilon[j]
                self.params['w'][2] = max_loss_weights

                prediction = self.forward(x)
                max_loss = self.loss(prediction, t, 1)

                # Calculate the finite gradient for the current N and current theta
                finite_gradient[j][i] = (max_loss - min_loss)/(2*epsilon[j])

        max_diff = np.max(np.abs((np.sum(true_gradient[2], axis=0) - finite_gradient)), axis=1)

        plt.plot(np.log(N), max_diff)
        plt.xlabel("Log(N)")
        plt.ylabel("Maximum difference between true and finite gradient")
        plt.show()

    def reset(self):
        self.params['w'] = []
        self.params['b'] = []
