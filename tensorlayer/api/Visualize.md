##Visualize model and data
####visualize model parameters
 tensorlayer.visualize.W(W=None, second=10, saveable=True, shape=[28,28], name='mnist', fig_idx=2398612)
	visualize every columns of the weight matrix to a group of greyscale img
parameters:
	W: numpy.array
	second: int
	saveable: boolean
	shape: a list with 2 int
		the shape of feature image.
	name: string
		a name to save the image, if saveable is True
	fig_idx: int
		matplotlib figure index

visualize cnn 2d filter:
 tensorlayer.visualize.CNN2d(CNN=None, second=10, saveable=True, name='CNN', fig_idx=3119362)
	display a group of RGB or Greyscale CNN masks
parameters:
	CNN: numpy.array
		the image. 64 5*5 RGB images can be (5,5,3,64)
	second: int
		the display second for the images, if saveable is false.
	saveable: boolean
	name: string
		a name to save the image, if saveable is True
	fig_idx: int
		matplotlib figure index

####visualize images
 tensorlayer.visualize.frame(I=None, second=5, saveable=True, name='frame', cmap=None, fig_idx=12836)
	display a frame. make sure OpenAI Gym render() is disable before using it.
parameters:
	I: numpy.array
		the image
	second: int 
		the display second for the image, if saveable is True
	saveable: boolean
		save or plot the figure.
	name: string
	cmap: None or string
	fig_idx: int
		matplotlib figure index.

 tensorlayer.visualize.images2d(images=None, second=10, saveable=True, name='images', dtype=None, fig_idx=3119362)
	display a group of RGB or Greyscale images.
parameters:
	dtype: None or numpy data type
		the data type for displaying the images

####visualize embeddings
 tensorlayer.visualize.tsne_embedding(embeddings, reverse_dictionary, plot_only=500, second=5, saveable=False, name='tsne',fig_idx=9862)
	visualize the embeddings by using t-SNE.
parameters:
	embedding: a matrix
	reverse_dictionary: a dictionary
		id_to_word, mapping id to unique word
	plot_only: int
		the number of examples to plot, choice the most common words.
	second: int 
		the display second for the images, if saveable is False.
	saveable: boolean
		save or plot the image
	name: a string
	fig_idx: int
		matplotlib figure index.
	
	
