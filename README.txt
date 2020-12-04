About:
	The data folder contains cleaned dataset.
	The src folder contains raw dataset, cleaned dataset, implimentation of neural networks using numpy and data preprocessing python source code.
	The data preprocessing i.e., Data_Handlers.py gets median for each column with result 0 and 1 seperatly and puts in empty columns.
	The implimentation of neural networks using numpy i.e., Neural_Net.py splits the dataset, initializes the model, trains model and predicts the test dataset.
	Neural network used:
		Their are 3 layers :
		    one input layer with 9 neurons, 
		    one hidden layer with 6 neurons,
		    one output layer with 2 neurons
		Loss function used is mean squared error (MSE) or L2 loss
		Learing rate is 0.1
		Number of epochs are 1450
		Activation function used is sigmoid
	The program can be modified accordingly to add more neurons in any layers.
	It can be modified for using relu as activation function.

How to run it:
	go to src folder run : python Neural_Net.py

