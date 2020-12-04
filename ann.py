'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''
import pandas as pd
import numpy as np

def data_preprocessing():
    # Read csv file into a pandas dataframe
    df = pd.read_csv("LBW_Dataset.csv")
#     z=24
#     df_1,df_2 = df.iloc[:z,:],df.iloc[z:,:]
#     for i in df_1.columns.tolist():
#         # Replace using median
#         median = df_1[i].median()
#         df_1[i].fillna(median, inplace=True)
#     for i in df_2.columns.tolist():
#         # Replace using median
#         median = df_2[i].median()
#         df_2[i].fillna(median, inplace=True)
#     df = pd.concat([df_1,df_2])
    for i in df.columns.tolist():
        # Replace using median
        median = df[i].median()
        df[i].fillna(median, inplace=True)
    # saving the dataframe
    df.to_csv('cleaned_LBW_Dataset.csv')

def data_splitting():
    x = np.genfromtxt('cleaned_LBW_Dataset.csv', delimiter=',')
    x = np.delete(x,0,axis=0)
    x = np.delete(x,0,axis=1)
    np.random.shuffle(x)
    training, test = x[:np.math.ceil(len(x)*0.7),:], x[np.math.ceil(len(x)*0.7):,:]
    x_train,y_train = training[:, :-1],training[:,-1]
    x_test,y_test = test[:, :-1],test[:,-1]
    return [x_train,y_train,x_test,y_test]

class NN:

    def __init__(self, input_dim=None, output_dim=None, hidden_layers=None, seed=1, activation_fun=0,eta=0.1, n_epochs=200):
        if (input_dim is None) or (output_dim is None) or (hidden_layers is None):
            raise Exception("Invalid arguments given!")
        self.input_dim = input_dim # number of input nodes
        self.output_dim = output_dim # number of output nodes
        self.hidden_layers = hidden_layers # number of hidden nodes @ each layer
        self.network = self._build_network(seed=seed)
        self.activation_fun = activation_fun # 0 for sigmoid 1 for relu
        self.eta = eta # eta value
        self.n_epochs = n_epochs # number of epochs

    ''' X and Y are dataframes '''

    def fit(self,X,Y):
        '''
        Function that trains the neural network by taking x_train and y_train samples as input
        '''
        for epoch in range(self.n_epochs):
            for (x_, y_) in zip(X, Y):
                self._forward_pass(x_) # forward pass (update node["output"])
                yhot_ = self._one_hot_encoding(y_, self.output_dim) # one-hot target
                self._backward_pass(yhot_) # backward pass error (update node["delta"])
                self._update_weights(x_,self.eta) # update weights (update node["weight"])

    def predict(self,X):

        """
        The predict function performs a simple feed forward of weights
        and outputs yhat values

        yhat is a list of the predicted value for df X
        """
        yhat = np.array([np.argmax(self._forward_pass(x_)) for x_ in X], dtype=np.int)
        return yhat

    def CM(self,y_test,y_test_obs):
        '''
        Prints confusion matrix
        y_test is list of y values in the test dataset
        y_test_obs is list of y values predicted by the model

        '''
        for i in range(len(y_test_obs)):
            if(y_test_obs[i]>0.6):
                y_test_obs[i]=1
            else:
                y_test_obs[i]=0

        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0

        for i in range(len(y_test)):
            if(y_test[i]==1 and y_test_obs[i]==1):
                tp=tp+1
            if(y_test[i]==0 and y_test_obs[i]==0):
                tn=tn+1
            if(y_test[i]==1 and y_test_obs[i]==0):
                fp=fp+1
            if(y_test[i]==0 and y_test_obs[i]==1):
                fn=fn+1
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp
   
        print("Confusion Matrix : ")
        print(cm)
        print("\n")
       
        p= tp/(tp+fp)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)

        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")

    # Build fully-connected neural network (no bias terms)
    def _build_network(self, seed=1):
        random.seed(seed)

        # Create a single fully-connected layer
        def _layer(input_dim, output_dim):
            layer = []
            for i in range(output_dim):
                weights = [random.random() for _ in range(input_dim)] # sample N(0,1)
                node = {"weights": weights, # list of weights
                        "output": None, # scalar
                        "delta": None} # scalar
                layer.append(node)
            return layer

        # Stack layers (input -> hidden -> output)
        network = []
        if len(self.hidden_layers) == 0:
            network.append(_layer(self.input_dim, self.output_dim))
        else:
            network.append(_layer(self.input_dim, self.hidden_layers[0]))
            for i in range(1, len(self.hidden_layers)):
                network.append(_layer(self.hidden_layers[i-1], self.hidden_layers[i]))
            network.append(_layer(self.hidden_layers[-1], self.output_dim))

        return network

    # Forward-pass (updates node['output'])
    def _forward_pass(self, x):
        if self.activation_fun == 0:
            transfer = self._sigmoid
        else:
            transfer = self._relu
        x_in = x
        for layer in self.network:
            x_out = []
            for node in layer:
                node['output'] = transfer(self._dotprod(node['weights'], x_in))
                x_out.append(node['output'])
            x_in = x_out # set output as next input
        return x_in

    # Backward-pass (updates node['delta'], L2 loss is assumed)
    def _backward_pass(self, yhot):
        if self.activation_fun == 0:
            transfer_derivative = self._sigmoid_derivative # sig' = f(sig)
        else:
            transfer_derivative = self._relu_derivative # relu' = f(relu)
        n_layers = len(self.network)
        for i in reversed(range(n_layers)): # traverse backwards
            if i == n_layers - 1:
                # Difference between logits and one-hot target
                for j, node in enumerate(self.network[i]):
                    err = node['output'] - yhot[j]
                    node['delta'] = err * transfer_derivative(node['output'])
            else:
                # Weighted sum of deltas from upper layer
                for j, node in enumerate(self.network[i]):
                    err = sum([node_['weights'][j] * node_['delta'] for node_ in self.network[i+1]])
                    node['delta'] = err * transfer_derivative(node['output'])

    # Update weights (updates node['weight'])
    def _update_weights(self, x, eta):
        for i, layer in enumerate(self.network):
            # Grab input values
            if i == 0:
                inputs = x
            else:
                inputs = [node_['output'] for node_ in self.network[i-1]]
            # Update weights
            for node in layer:
                for j, input in enumerate(inputs):
                    # dw = - learning_rate * (error * transfer') * input
                    node['weights'][j] += - eta * node['delta'] * input

    # Dot product
    def _dotprod(self, a, b):
        return sum([a_ * b_ for (a_, b_) in zip(a, b)])

    # Sigmoid (activation function)
    def _sigmoid(self, x):
        return 1.0/(1.0+math.exp(-x))

    # Sigmoid derivative
    def _sigmoid_derivative(self, sigmoid):
        return sigmoid*(1.0-sigmoid)

    # Relu (activation function)
    def _relu(self,x):
        return np.maximum(0,x)

    # Relu derivative
    def _relu_derivative(self, x):
        return 1 * (x > 0)

    # One-hot encoding
    def _one_hot_encoding(self, idx, output_dim):
        x = np.zeros(output_dim, dtype=np.int)
        x[int(np.ceil(idx))] = 1
        return x

data_preprocessing()

d = 9
n_classes = 2
hidden_layers = [6,6,6,6,2] # number of nodes in hidden layers i.e. [layer1, layer2, ...]
eta = 0.001 # learning rate
n_epochs = 100 # number of training epochs
seed_weights = 1 # seed for NN weight initialization
activation_function = 0 # activation function used is relu

x_train,y_train,x_test,y_test = data_splitting()
model = NN(input_dim=d, output_dim=n_classes, hidden_layers=hidden_layers, seed=seed_weights, activation_fun=activation_function, eta=eta, n_epochs=n_epochs)
# print(x_train,y_train)
model.fit(x_train, y_train)

# Make predictions for training and test data
ypred_train = model.predict(x_train)
ypred_test = model.predict(x_test)
# print(ypred_train,y_train)
# print(ypred_test,y_test)

# print(list(y_test),list(ypred_test))
model.CM(y_test,ypred_test)
model.CM(y_train,ypred_train)
# print(list(y_train),list(ypred_train))
