class Model(object):
    def load_data(self):
        raise NotImplementedError("")

    def add_placeholders(self):
        #for inputs 
        raise NotImplementedError("")

    def create_feed_dict(self,input_batch,label_batch):
        #create the feed_dict for training the given step.
        raise NotImplementedError("")

    def add_model(self, input_data):
        #implements core of model that transforms input_data into predictions
        
        raise NotImplementedError("")

    def add_loss_op(self,pred):
        # adds ops for loss to the computational graph.

        raise NotImplementedError("")

    def run_epoch(self,sess,input_data,input_labels):
        # runs an epoch of training , trains the model for one epoch.
        
        raise NotImplementedError("")

    def fit(self,sess,input_data, input_labels):
        # fit model on provided data

        raise NotImplementedError("")
    
    def predict(self,sess,input_data,input_labels=None):
        #make predictions from the provided model

        raise NotImplementedError("")

class LanguageModel(Momel):
    #Abstracts a tensorflow graph for learning language models, add ability to do embedding.

    def add_embedding(self):

        raise NotImplementedError("")


