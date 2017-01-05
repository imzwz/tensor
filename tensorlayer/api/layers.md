##Layers
####Basic layer
 layer.outputs: a Tensor, the outputs of current layer.
 layer.all_params: a list of Tensor, all network variables in order.
 layer.all_layers: a list of Tensor, all network outputs in order.
 layer.all_drop: a dictionary of {placeholder: float}, all keeping probalilities of noise layer.
共有的方法(common method):
 layer.print_params(): print the network variables information in order.或者使用tl.layers.print_all_variables().
 layer.print_layers(): print the network layers information in order.
 layer.count_params(): print the number of parameters in the network.
 exp: all_params = [W1, b1, W2, b2, W_out, b_out] //定义一个三层网络
使用network.all_params[2:3]获取或者使用get_variables_with_name()来获取param

####Dense layer
 tensorlayer.layers.DenseLayer(layer=None, n_units=100, act=<function identity>, W_init=<function _initializer>, b_init=<funciton _initilaizer>, W_init_args={}, b_init_args={}, name='dense_layer')
Dense layer ist a fully connected layer.
params:
	layer: a Layer instance
	n_units: int  该层的units数
	act: activation function 激活函数
	W_init: weithts initializer W的初始值
	b_init: biases initializer b的初始值
	W_init_args: dictionary 使用tf.get_variable获得的参数
	b_init_args: dictionary 使用tf.get_variable获得的参数
	name: string or none 该层的名字

####Custom layer
自定义层需要实现Layer层，例如：
~~~
class DoubleLayer(Layer):
	def __init__(
		self,
		layer = None,
		name = 'double_layer',
	):
		Layer.__init__(self, name=name)
		self.inputs = layer.outputs
		self.outputs = self.inputs * 2
		self.all_layers = list(layer.all_layers)
		self.all_params = list(layer.all_params)
		self.all_drop = dict(layer.all_drop)
		self.all_layers.extend([self.outputs])
~~~
上面的例子实现了一个层将它的输入乘以2

####Name Scope and Sharing Parameters
 tensorlayer.layers.get_variables_with_name(name, train_only=True, printable=False) 通过指定名字获取变量列表
 tensorlayer.layers.set_name_reuse(enable=True)复用变量名
 tensorlayer.layers.print_all_variable(train_only=False) 输出所有变量，train_only为True时只输出trainable变量 

####Input layer
InputLayer(inputs=None, n_features=None, name='input_layer') 是神经网络中的开始层
params: 
	inputs: a TensorFlow placeholder  输入的tensor数据
	name: a string or None 输入层的名称
	n_features: a int   the number of features

####Word Embedding input
 tensorlayer.layers.Word2vecEmbeddingInputlayer(inputs=None, train_labels=None, vocabulary_size=80000, embedding_size=200, num_sampled=64, nce_loss_args={}, E_init=<function _initializer>, E_init_args={}, nce_W_init=<function _initializer>, nce_b_init=<function _initializer>, nce_b_init_args={}, name='word2vec_layer')
 Word2vecEmbeddingInputlayer 是一个用于词嵌入的全连接层，输入是integer index，输出是词向量
 params:
	inputs: placeholder 输入，integer index格式
	train_labels: placeholder 标签，integer index格式
	vocabulary_size: int 词汇量大小
	embedding_size: int 词向量维数
	num_sampled: int 负向抽样数目
	nce_loss_args: a dictionary tf.nn.nce_loss()所返回的参数
	E_init: embedding initializer 词向量初始化
	E_init_args: a dictionary 词向量初始化参数
	nce_W_init: NCE decoder biases initializer NCE解码器weight初始化
	nce_W_init_args: a dictionary NCE解码器weight初始化参数
	nce_b_init: NCE decoder biases initializer NCE解码器biase初始化
	nce_b_init_args: a dictinoary NCE解码器biase初始化参数
	name: a string or None 该层名称

 tensorlayer.layers.EmbeddingInputlayer(inputs=None, vocabulary_size=80000, embedding_size=200, E_init=<function _intializer>, E_init_args={}, name='embedding_layer')
可以直接使用word2vec训练得到的word vector


