import mxnet as mx

def YOLO_loss(predict, label):
	# reshape input to desired shape
	"""
	predict(param) : mx.sym ->which is NDArray(tensor) its shape is (batch_size, 7,7,5)
	lable: softsign = x/1+|x|

	"""
	predict = mx.sym.reshape(predict, shape=(-1, 49, 5))
	
	#shift everything to (0, 1)
	predict_shift = (predict + 1) / 2
	label = mx.sym.reshape(label, shape=(-1, 49, 5))
	
	# split the tensor the order of (prob, x, y, w, h)
	cl, xl, yl, wl, hl = mx.sym.split(label, num_outputs=5, axis=2)
	cp, xp, yp, wp, hp = mx.sym.split(predict_shift, num_outputs=5, axis=2)

	
	lambda_coord = 5
	lambda_obj = 1
	lambda_noobj = 0.2
	# cl = 0 -> mask = 0.2
	# cl = 1 -> mask = 1
	mask = cl * lambda_obj + (1-cl) * lambda_noobj
	lossc = mx.sym.LinearRegressionOutput(label=cl*mask, data=cp*mask)
	lossx = mx.sym.LinearRegressionOutput(label=cl*lambda_coord*xl, data=cl*lambda_coord*xp)
	lossy = mx.sym.LinearRegressionOutput(label=cl*lambda_coord*yl, data=cl*lambda_coord*yp)
	lossw = mx.sym.LinearRegressionOutput(label=cl*lambda_coord*mx.sym.sqrt(wl), 
										  data=cl*lambda_coord*mx.sym.sqrt(wp))
	lossh = mx.sym.LinearRegressionOutput(label=cl*lambda_coord*mx.sym.sqrt(hl), 
										  data=cl*lambda_coord*mx.sym.sqrt(hp))
	loss = lossc+lossx+lossy+lossw+lossh
	return loss

# Get pretrained imagnet model
def get_resnet_model(model_path, epoch):
	label = mx.sym.Variable('softmax_label')
	# load symbol and actual weights
	sym, args, aux = mx.model.load_checkpoint(model_path, epoch)

	# extract last bn layer
	sym = sym.get_internals()['bn1_output']

	# append two layers
	sym = mx.sym.Activation(data = sym, act_type = "relu")
	sym = mx.sym.Convolution(data=sym, kernel = (3, 3),
		num_filter=5, pad=(1, 1), stride=(1, 1), no_bias=True,)
    
	# get softsign
	sym = sym / (1 + mx.sym.abs(sym))
	logit = mx.sym.transpose(sym, axes=(0, 2, 3, 1), name="logit")

	#apply loss
	loss_  = YOLO_loss(logit, label)

	# mxnet special requirement
	loss = mx.sym.MakeLoss(loss_)
	# multi-output logit should be blocked from generating gradients
	out = mx.sym.Group([mx.sym.BlockGrad(logit), loss])
	return out
