##Iteration
data iteration
####Non-time series
tensorlayer.iterate.minibatches(inputs=None, targets=None, batch_size=None, shuffle=False)

####Time series
tensorlayer.iterate.seq_minibatches(inputs, targets, batch_size, seq_length, stride=1)
tensorlayer.iterate.seq_minibatches2(inputs, targets, batch_size, num_steps)

####PTB dataset iteration
tensorlayer.iterate.ptb_iterator(raw_data, batch_size, num_steps)

