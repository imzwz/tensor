##Iteration
data iteration
####Non-time series
 tensorlayer.iterate.minibatches(inputs=None, targets=None, batch_size=None, shuffle=False)
parameters:
	inputs: numpy.array
		the input features, every row is a example.
	targets: numpy.array
		the labels of inputs, every row is a example.
	batch_size: int 
		batch size
	shuffle: boolean
		是否使用随机队列，打乱返回的数据集顺序
~~~
x = np.asarray([['a','a'], ['b','b'],['c','c'],['d','d'],['e','e'],['f','f']])
y = np.asarray([0,1,2,3,4,5])
for batch in tl.iterate.minibatches(inputs=x, targets=y, batch_size=2, shuffle=False):
	print(batch)

~~~

####Time series
 tensorlayer.iterate.seq_minibatches(inputs, targets, batch_size, seq_length, stride=1)
	生成一个返回batch_size*seq_length的batch

 tensorlayer.iterate.seq_minibatches2(inputs, targets, batch_size, num_steps)
praramters:
	inputs: a list
	targets: a list
	batch_size: int
	num_steps: int
		the number of unrolls
生成一对batch数据，每一对都是[batch_size, num_steps]的矩阵


####PTB dataset iteration
 tensorlayer.iterate.ptb_iterator(raw_data, batch_size, num_steps)
parameters:
	raw_data: a list
	batch_size: int
	num_steps: int
		the number of unrolls


