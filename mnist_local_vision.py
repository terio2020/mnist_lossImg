import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import numpy as np

# 网络自动下载mnist数据集
# mnist = tf.keras.datasets.mnist
# 直接对mnist数据集进行加载
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# 手动加载mnist数据集
datapath  = r'./mnist.npz'
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data(datapath)
x_test1 = x_test    #用于打印图片28*28,不能是28*28*1，第三维度只能是3或4
x_train = x_train.reshape(x_train.shape[0],784).astype('float32')
x_test = x_test.reshape(x_test.shape[0],784).astype('float32')


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)   #归一化

# 创建网络模型
model = Sequential([ # 3 个非线性层的嵌套模型
    layers.Flatten(),  #将多维数据打平,也即reshape为60000*784，否则就要在前面先reshape，而reshape之后就不用这个了
    layers.Dense(500, activation='relu'),#128
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 输出模型结构
model.build((None,784,1))
print(model.summary())

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
# model.fit(x_train, y_train, epochs=2,verbose=2) #verbose为1表示显示训练过程
                                                #x_train是二维数组[图片数量，图片大小]; y_train==>[图片数量，10个分类]


t = model.fit(x_train, y_train, epochs=2,verbose=2)
tl = t.history['loss']            # fit是自动进行学习，计算loss、acc并更新，这样可以提取出其中的loss或者acc
print("train_loss:", tl[1])


# 将训练的模型计算出准确率和损失率  方式一
# val_loss, val_acc = model.evaluate(x_test, y_test) # model.evaluate是输出计算的损失和精确度
# print('First Test Loss:{:.6f}'.format(val_loss))
# print('First Test Acc:{:.6f}'.format(val_acc))


# 将训练的模型计算出准确率 方式二
# acc_correct = 0
# predictions = model.predict(x_test)     # 将测试集用模型进行预测
# for i in range(len(x_test)):
#     if (np.argmax(predictions[i]) == y_test[i]):    # argmax是取最大数的索引,放这里是最可能的预测结果
#         acc_correct += 1
# print('Test accuracy:{:.6f}'.format(acc_correct*1.0/(len(x_test))))     #用测试集得到的模型准确率


# 保存训练后的模型
# model.save('epic_num_reader.model')
# new_model = tf.keras.models.load_model('epic_num_reader.model')


# 调用保存后的模型
# predictions = new_model.predict([x_test])       #调用
# print(predictions[0])                           #输出第一张图的预测情况
# print(np.argmax(predictions[0]))                #输出预测的第一张图的数字是多少

#绘制前25张图参观
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(x_test1[i], cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
plt.show()
