class NN(object):
    def __init__(self):
        """ init the model with hyper-parameters etc """
        self.numLayers = 3
        self.layersNeurans = [3,3,3]
        self.Weights = np.
        
    def inference(self, x):
        """ This is the forward calculation from x to y """
        
        return some_op(x, name="inference")
    
    def loss(self, batch_x, batch_y=None):
        y_predict = self.inference(batch_x)
        self.loss = tf.loss_function(y, y_predict, name="loss") # supervised
        # loss = tf.loss_function(x, y_predicted) # unsupervised
        
    def optimize(self, batch_x, batch_y):
        return tf.train.optimizer.minimize(self.loss, name="optimizer")
    
