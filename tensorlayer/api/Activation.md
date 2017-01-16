##tensorlayer activation
####identity
 tensorlayer.activation.identity(x, name=None)
parameters: 
	x: a tensor input
	return a tensor with the same type as x

####Ramp
 tensorlayer.activation.ramp(x=None, v_min=0, v_max=1, name=None)
	the ramp activation function
parameters:
	x: a tensor input
	v_min: float
	v_max: float
	name: a string or None
	returns a tensor with the same type as x

####Leaky Relu
 tensorlayer.activation.leaky_relu(x=None, alpha=0.1, name='LeakyReLU')
	the leakyrelu, shortcut is lrelu, introducing a nonzero gradient for negative input
parameters:
	x: a tensor with type float, double, int32, int64, unit8, int16, or int8
	alpha: float. slope
	name: string

####pixel-wise softmax
 tensorlayer.activation.pixel_wise_softmax(output, name='pixel_wise_softmax')
	return the softmax outputs of images, every pixels have multiple label, the sum of a pixel is 1. Usually be used for image segmentation.




		
