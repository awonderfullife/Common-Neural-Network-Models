import mxnet as mx
import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

mnist = mx.test_utils.get_mnist()

class Classifier:
    def __init__(self, batch_size=128, learning_rate=0.01, data=mnist):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_iter = mx.io.NDArrayIter(data['train_data'], data['train_label'], batch_size, shuffle=True)
        self.val_iter = mx.io.NDArrayIter(data['test_data'], data['test_label'], batch_size)
        predict = self.build_model()
        self.mx_model = mx.mod.Module(symbol=predict, context=mx.gpu())

    def build_model(self):
        input_layer = mx.sym.var('data')

        conv_layer1 = mx.sym.Convolution(data=input_layer, kernel=(5,5), num_filter=20)
        actv_layer1 = mx.sym.Activation(data=conv_layer1, act_type="relu")
        pool_layer1 = mx.sym.Pooling(data=actv_layer1, pool_type="max", kernel=(2,2), stride=(2,2))

        conv_layer2 = mx.sym.Convolution(data=pool_layer1, kernel=(5,5), num_filter=50)
        actv_layer2 = mx.sym.Activation(data=conv_layer2, act_type="relu")
        pool_layer2 = mx.sym.Pooling(data=actv_layer2, pool_type="max", kernel=(2,2), stride=(2,2))

        flatten_layer1 = mx.sym.flatten(data=pool_layer2)

        fc_layer1 = mx.sym.FullyConnected(data=flatten_layer1, num_hidden=64)
        actv_layer3 = mx.sym.Activation(data=fc_layer1, act_type="relu")

        fc_layer2 = mx.sym.FullyConnected(data=actv_layer3, num_hidden=10)
        predict = mx.sym.SoftmaxOutput(data=fc_layer2, name="softmax")

        return predict
    
    def train(self):
        self.mx_model.fit(
            self.train_iter,
            self.val_iter,
            optimizer="sgd",
            optimizer_params={"learning_rate":0.1},
            eval_metric='acc',  # report variable
            batch_end_callback=mx.callback.Speedometer(self.batch_size, 100), # report each 100 batch_size
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




