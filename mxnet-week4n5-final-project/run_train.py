import mxnet as mx
from symbol import get_resnet_model
import numpy as np 
from data_ulti import get_iterator
#import matplotlib.pyplot as plt

import logging
import sys
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    # get sym
    # Try different network 34 50 101 to find the best fit
    # 1. Get pretrained symbol
    sym = get_resnet_model('pretrained_models/resnet-34', 0)
    
    _, args_params, aux_params = mx.model.load_checkpoint('pretrained_models/resnet-18', 0)
    
    # get some input
    train_data = get_iterator(path='DATA_rec/cat_full.rec', data_shape=(3, 224,224),
                             label_width=7*7*5, batch_size = 10, shuffle = True)
    val_data = get_iterator(path='Data_rec/cat_val.rec', data_shape=(3,224,224),
                             label_width=7*7*5, batch_size = 10)
    # allocate memory to the system
    mod = mx.mod.Module(symbol=sym, context=mx.gpu(0))
    
    precision_list = []
    recall_list = []
    
    def loss_metric(label, pred):
        """
        label: np.array->(batch_size, 7,7,5)
        pre: same as label
        """
        # parse label, pred
        label = label.reshape((-1, 7, 7, 5)) #0<all<1
        pred = pred.reshape((-1, 7, 7, 5))
        # from (-1, 1) -> (0, 1)
        pred_shift = (pred+1)/2
        
        # Revert to original size
        """ def decodeBox(yolobox, size=224, dscale=32):
            i, j, x, y, w, h = yolobox
            cxt = i*dscale + x*dscale
            cyt = j*dscale + y*dscale
            wt = w*size
            ht = h*size
            return [cxt, cyt, wt, ht] """
        cl = label[:, :, :, 0] #prob: no need to upscale 
        x1 = label[:, :, :, 1]*32
        yl = label[:, :, :, 2]*32
        w1 = label[:, :, :, 3]*224
        h1 = label[:, :, :, 4]*224
        cp = pred_shift[:, :, :, 0]
        xp = pred_shift[:, :, :, 1]*32
        yp = pred_shift[:, :, :, 2]*32
        wp = pred_shift[:, :, :, 3]*224
        hp = pred_shift[:, :, :, 4]*224
        
        # number of boxes
        num_box = np.sum(cl)
        # cp < 0.5 -> predict negative -> will return 1
        # cl -> positive -> 1
        # calculate how may 1
        FN = np.sum(cl * (cp < 0.5) == 1)
        # cl -> 0
        # cp -> 1
        FP = np.sum((1 - cl) * (cp > 0.5))
        TP = np.sum(cl * (cp > 0.5) == 1)
        #recall = TP / (TP + FN)
        recall = np.sum(cl * (cp > 0.5) == 1) / num_box
        #precision = TP / (TP + FP)
        precision =  np.sum(cl * (cp > 0.5) == 1) / (np.sum(cp > 0.5) + 2e-5)
        
        F1_score = 2 * precision * recall / (precision + recall)
        
        #add precison and recall to lists
        #precision_list.append(precision)
        #recall_list.append(recall)
    
        print("Recall is {}".format(recall))
        print("Precision is {}".format(precision))
        print("F1 score is {}".format(F1_score))
        print("Number of FN is {}".format(FN))
        print("Number of FP is {}".format(FP))
        print("The total number of boxes is {}".format(num_box))
        print("FN boxes : {}".format(np.where(cl*(cp<0.5) == 1)))
    
        
        return -1;
        
    # setup metric
    metric = mx.metric.create(loss_metric, allow_extra_outputs=True)
    
    # setup monitor for debugging
    def norm_stat(d):
        return mx.nd.norm(d) / np.sqrt(d.size)
    mon = None
    
   
    # Reference: http://mxnet.io/api/python/callback.html
    # A callback that saves a model checkpoint every few epochs.   
    checkpoint = mx.callback.do_checkpoint('cat_full_model', 1)
    
    # Train
    # Try different hyperparamters: batch_size, optimization 
   # epoch = range(0, 2000)
    mod.fit(train_data=train_data,
            eval_data =val_data,
            num_epoch =1000,
            monitor=mon,
            eval_metric=[metric],
            optimizer='Adagrad',
            optimizer_params={'learning_rate': 0.001, 'lr_scheduler':
                            mx.lr_scheduler.FactorScheduler(300000, 0.10, 0.0001)},
            initializer=mx.init.Xavier(magnitude=2, rnd_type='gaussian', factor_type='in'), #for missing values
            arg_params=args_params,
            aux_params=aux_params,
            allow_missing = True,
            # things to do after training of one batch
            batch_end_callback=[mx.callback.Speedometer(batch_size=10, frequent=10, auto_reset=False)],
            # things to do after an epoch
            epoch_end_callback=checkpoint,
           )
#     plt.plot(epoch, precision_list, "r")
#     plt.plot(epoch, recall_list, "b")
#     plt.show()

