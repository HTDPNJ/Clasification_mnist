import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import RMSprop

#下载数据集
(X_train,y_train),(X_test,y_test)=mnist.load_data()

X_train=X_train.reshape(X_train.shape[0],-1) /255  #标准化
X_test=X_test.reshape(X_test.shape[0],-1)/255
y_train=np_utils.to_categorical(y_train,num_classes=10)
y_test=np_utils.to_categorical(y_test,num_classes=10)  #搞成0，1编码

model=Sequential([
    Dense(output_dim=32,input_dim=28*28),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

rmaprop=RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)  #优化器，更新数据lr是学习率

model.compile(
    #optimizer='rmaprop' #默认的
    optimizer=rmaprop,
    loss='categorical_crossentropy',
    metrics=['accuracy'], #训练的时候可以计算一下
)



import numpy
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score


class Metrics(Callback):     #修改一下，使用召回率，f1什么的进行评估
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (numpy.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return
metrics = Metrics()
model.fit(X_train, y_train, epochs=2, batch_size=32,
                      verbose=0, validation_data=(X_test, y_test),
                      callbacks=[metrics])

print("\ntesting---------")
f1=metrics.val_f1s
print('test loss',f1)
precisions=metrics.val_precisions
print('\ntest loss',precisions)
recalls=metrics.val_recalls
print('\ntest loss',recalls)
print("training-----")
#model.fit(X_train,y_train,nb_epoch=2,batch_size=32)   #用系统默认的进行评估
#loss,accuracy=model.evaluate(X_test,y_test)
#print('test loss',loss)
#print('test accuracy',accuracy)