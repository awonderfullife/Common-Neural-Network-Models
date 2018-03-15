import mxnet as mx
import logging

mnist = mx.test_utils.get_mnist()
logging.getLogger().setLevel(logging.DEBUG)

class Classifier:
    def __init__(self, batch_size=128, learning_rate=0.01, data=mnist):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_iter = mx.io.NDArrayIter(data['train_data'], data['train_label'], batch_size, shuffle=True)
        self.val_iter = mx.io.NDArrayIter(data['test_data'], data['test_label'], batch_size)
        predict = self.build_model()
        self.mx_model = mx.mod.Module(symbol=predict, context=mx.gpu())


    def build_model(self):
        input_data = mx.sym.var('data')
        input_layer = mx.sym.flatten(data=input_data)

        fc_layer1 = mx.sym.FullyConnected(data=input_layer, num_hidden=128)
        ac1 = mx.sym.Activation(data=fc_layer1, act_type="relu")

        fc_layer2 = mx.sym.FullyConnected(data=ac1, num_hidden=64)
        ac2 = mx.sym.Activation(data=fc_layer2, act_type="relu")

        fc_layer3 = mx.sym.FullyConnected(data=ac2, num_hidden=10)
        predict = mx.sym.SoftmaxOutput(data=fc_layer3, name="softmax")

        return predict
    
    def train(self):
        self.mx_model.fit(
            train_data=self.train_iter,
            eval_data=self.val_iter,
            optimizer="sgd",
            optimizer_params={"learning_rate":0.1},
            eval_metric='acc',  # report variable
            batch_end_callback=mx.callback.Speedometer(self.batch_size, 100), # log each 100 batch_size
            num_epoch=10
        )
    
    def test(self):
        acc = mx.metric.Accuracy()
        self.mx_model.score(self.val_iter, acc)
        print acc

if __name__ == '__main__':
    classfier = Classifier()
    classfier.train()
    classfier.test()




