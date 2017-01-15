##Load save model and data
####Load dataset function
mnist:	
 tensorlayer.files.load_mnist_dataset(shape=(-1,784))

cifar-10:
 tensorlayer.files.load_cifar10_dataset(shape=(-1,32,32,3), plotable=False, second=3)
parameters:
	shape: tupe
		the shape of digit images
	plotable: boolean
		whether to ploat some image examples
	second: int 
		if plotable is True, second is the display time.

penn treebank(ptb):
 tensorlayer.files.load_ptb_dataset()
	return train_data, valid_data, test_data, vocab_size

imbd:
 tensorlayer.files.load_imbd_dataset(path='imdb.pkl', nb_words=None, skip_top=(), maxlen=None, test_split=0.2, seed=113, start_char=1, oov_char=2, index_from=3)

Nietzsche:
 tensorlayer.files.load_nietzsche_dataset()

####Load and save network
Save network as .npz:
 tensorlayer.files.save_npz(save_list=[], name='model.npz', sess=None)

Load network from .npz:
 tensorlayer.files.load_npz(path='', name='model.npz')
	return a list of parameters in order

Assign parameters to network:
 tensorlayer.files.assign_params(sess, params, network)
parameters:
	sess: tensorflow session
	params: a list
		a list of parameters in order
	network: a layer class
		the network to be assigned

####Load and save variables
Save variables as .npy:
 tensorlayer.files.save_any_to_npy(save_dict={}, name='any.npy')

Load variables from .npy:
 tensorlayer.files.load_npy_to_any(path='', name='any.npy')

####visualizing npz file
 tensorlayer.files.npz_to_W_pdf(path=None, regx='wlpre_[0-9]+\.(npz)')
parameters:
	regx: a string, regx for the file name

####helper functions
 tensorlayer.files.load_file_list(path=None, regx='\.npz', printable=True)
	return a file list in a folder by given a path and regular expression.



