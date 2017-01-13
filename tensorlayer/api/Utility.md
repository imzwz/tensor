##Utility
####Training
 tensorlayer.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_, acc=None, batch_size=100, n_epoch=100, print_freq=5, X_val=None, y_val=None, eval_train=True)
parameters:
	sess: Tensorflow session
	network: a Tensorlayer layer
	train_op: a Tensorflow optimizer
	X_train: numpy array
		input of training data
	y_train: numpy array
		the target of training data
	x: placeholder
	y_: placeholder
	acc: the tensorflow expressiong of accuracy or None
		if None, would not display the metric
	batch_size: int 
	n_epoch: int
		the number of training epochs
	print_freq: int
		display the training information every print_freq epochs
	X_val: numpy array or None
		the input of validation data
	y_val: numpy array or None
		the target of validation data
	eval_train: boolean
		if X_val and y_val are not None, if refects whether to evaluate the training data

####Evaluation
 tensorlayer.utils.test(sess, network, ac, X_test, y_test, x, y_, batch_size, cost=None)
parameters:
	sess: tensorflow session
	network: a tensorlayer layer
	acc: the tensorflow expression of accuracy or None
		if None, would not display the metric
	X_test: numpy array
		the input of test data
	y_test: numpy array
		the target of test data
	x: placeholder
	y_: placeholder
	batch_size: int or None
		batch size for testing, when dataset is large, we should use minibatche for testing, when dataset is small, we can set it to None
	cost: the tensorflow expression of cost or None
		if None, would not display the cost

 tensorlayer.utils.evaluation(y_test=None, y_predict=None, n_classes=None)
	input the predicted results, targets results and the number of class, return the confusion matrix, F1-score of each class, accuracy and macro F1-score.
parameters:
	y_test: numpy.array or list
		target results
	y_predict: numpy.array or list
		predicted results
	n_classes: int
		number of classes

####Prediction
 tensorlayer.utils.predict(sess, network, X, x, y_op)
	return the precict result
parameters:
	sess: tensorflow session
	network: a tensorlayer layer
	X: numpy array
		the input
	y_op: placeholder
		the argmax expression of softmax outputs

####Class balancing
 tensorlayer.utils.class_balancing_oversample(X_train=None, y_train=None, printable=True)
	input the features and labels, return the features and labels after oversampling
parameters:
	X_train: numpy array
		Features, each row is an example
	y_train: numpy array
		labels

####helper functions
 tensorlayer.utils.dict_to_one(dp_dict={})
	input a dictionary, return a dictionary that all items are set to one, use for disable dropout, dropconnect layer and so on

 tensorlayer.utils.flatten_list(list_of_list=[[],[]])
	input a list of list, return a list that all items are in a list.



