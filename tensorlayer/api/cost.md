##Cost
####Cross entropy
 tensorlayer.cost.cross_entropy(output, target, name='cross_entropy_loss')
 exp: 
	ce = tl.cost.cross_entropy(y_logits, y_target_logits)

####Binary cross entropy
 tensorlayer.cost.binary_cross_entropy(output, target, epsilon=1e-08, name='bce_loss')
 ##loss(x,z) = -sum_i(x[i]*log(z[i])+(1-x[i])*log(1-z[i]))

####Mean squared error
 tensorlayer.cost.mean_squared_error(output, target)

####Dice coefficient
 tensorlayer.cost.dice_coe(output, target, epsilon=1e-10)

####Regularization functions

