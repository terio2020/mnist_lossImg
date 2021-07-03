import  tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import  datetime
from    matplotlib import pyplot as plt
import  io

# 对数据集预处理，转换数据类型
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x,y

# 将数据集的图片读取、转换成PNG格式并返回
def plot_to_image(figure):
  buf = io.BytesIO()     #在内存中读写bytes
  plt.savefig(buf, format='png')       #保存生成的图片
  plt.close(figure)
  buf.seek(0)       #文件读取指针移动到文件第“0”个字节处，这里即移动到文件头位置
  image = tf.image.decode_png(buf.getvalue(), channels=4)     #对PNG图像进行解码，得到图像的像素值，用于显示图像
  image = tf.expand_dims(image, 0)    #指定在第“0”维添加维数
  return image

# 将上面返回的PNG图像显示出来
def image_grid(images):
  figure = plt.figure(figsize=(10,10))   #创建宽和高分别为10英寸的图像
  for i in range(64):
    plt.subplot(8, 8, i + 1, title='name')  #生成36个子图像
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap=plt.cm.binary)  #将一个image显示在二维坐标轴上
  return figure

# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————
batchsz = 128       # 每次处理128张图

# 在线下载，载入mnist数据集（db是训练集，ds_val是测试集）
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('datasets:', x.shape, y.shape, x.min(), x.max())
db = tf.data.Dataset.from_tensor_slices((x,y))      #给定的元组、列表和张量等数据进行特征切片，输出结果的意义
                                                    # 是每x个特征对应y个标签。这里是将图片和标签分开，方便后处理
# print(db)                                         #如x=(6,3),y=(6,1) ==> db=((3,),(1,))
db = db.map(preprocess).shuffle(60000).batch(batchsz).repeat(10)    #map(preprocess) 调用preprocess函数，返回包含
                                                                    # 每次函数返回值的新列表
                                                                    #shuffle() 将数据打乱，数值越大，混乱程度越大
                                                                    #batch(batchsz) 按照顺序取出batchsz行数据
                                                                    #repeat(10) 重复10次
                                                                    #这里是将图片和标签进行10次分批次地打乱、类型转换
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))         # 上面是测试集，这里是训练集，重复上面的操作
ds_val = ds_val.map(preprocess).batch(batchsz, drop_remainder=True) #drop_remaindar，最后一个batch数据量达不到时是否保留

# 搭建了5层神经网络，之后调用network来处理图片数据
network = Sequential([layers.Dense(256, activation='relu'), #Squential将网络层和激活函数结合起来，输出激活后的网络节点
                     layers.Dense(128, activation='relu'),  #Dense全连接层，添加一个层；activation激活函数，非线性变化
                     layers.Dense(64, activation='relu'),
                     layers.Dense(32, activation='relu'),
                     layers.Dense(10)])
network.build(input_shape=(None, 28*28))   #build中存放本层需要初始化的变量；build()：在call()函数第一次执行时会被调用
network.summary()   #tensorboard读取网络训练过程中保存到本地的日志文件实现数据可视化，日志数据保存用到 tf.summary中的方法

optimizer = optimizers.Adam(lr=0.01)    #Adam优化器，学习率=0.01


# 获取当前时间，用于生成文件夹名以及路径，最后一句是实现保存日志这个动作
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  #strftime() 用来格式化datetime 对象
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)         # 创建并返回一个 SummaryWriter对象，
                                                                # 生成的日志将储存到logdir指定的路径中

# 从训练集中拿出图像集x，再取图像集中的第一张图像用来显示
sample_img = next(iter(db))[0]      #通过iter()函数获取对象list->db的迭代器,然后迭代器不断用next()函数来获取下⼀条数据
sample_img = sample_img[0]
sample_img = tf.reshape(sample_img, [1, 28, 28, 1])     #[batch_size,height, width, channels]
with summary_writer.as_default():  #创建一个默认会话，当上下文管理器退出时会话没有关闭，可以调用会话进行run()和eval()操作
    tf.summary.image("Training sample:", sample_img, step=0) #输出带图像的probuf，汇总数据的图像的的形式如下：
                                                                # ' tag /image/0',' tag /image/1'...如：input/image/0等

# 对训练集进行梯度下降运算
for step, (x,y) in enumerate(db):
    with tf.GradientTape() as tape:
        x = tf.reshape(x, (-1, 28*28))
        out = network(x)
        y_onehot = tf.one_hot(y, depth=10)
        loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, out, from_logits=True))
                                    #tf.reduce_mean（a，axis）是均值，其中a是输入矩阵，axis是从什么维度求均值
                                    #categorical_crossentropy计算交叉熵，交叉熵越小，预测值与真实值之间差异越小
    grads = tape.gradient(loss, network.trainable_variables) #g.gradient(y,x)==>计算dy/dx并代入x的指，结果是一个常数
                                                            # 自动更新梯度
    optimizer.apply_gradients(zip(grads, network.trainable_variables))
                                    #zip()将可迭代的对象中对应的元素打包成一个个元组，然后返回由元组组成的列表
                                    #grads_and_vars的格式是compute_gradients()所返回的(gradient, variable)
                                                            # 自动更新参数
    if step % 100 == 0:
        print(step, 'loss:', float(loss))   #画loss或accuary时显示标量信息
        with summary_writer.as_default():
            tf.summary.scalar('train-loss', float(loss), step=step) #summary.scalar(tags,values)

    # 用测试集评估模型的准确率
    if step % 500 == 0:
        total, total_correct = 0., 0
        for _, (x, y) in enumerate(ds_val):
            x = tf.reshape(x, (-1, 28*28))
            out = network(x)
            pred = tf.argmax(out, axis=1)   #tf.argmax(input,axis)返回最大值的索引。axis=0——>列；axis=1——>行
            pred = tf.cast(pred, dtype=tf.int32)    #将'pred'的数据格式转化成dtype数据类型
            correct = tf.equal(pred, y)         #比较两个矩阵或者向量的每一个元素，相应每个位置输出True or False
            total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy() #计算张量tensor沿着某一维度的和
                                                # tf.XXX.numpy() 转换tensor为numpy array
            total += x.shape[0]
        print(step, 'Evaluate Acc:', total_correct/total)

        # 输出36张训练集的图片
        val_images = x[:64] #取36张图片
        val_images = tf.reshape(val_images, [-1, 28, 28, 1])    #将36张所有图片转换形状
        with summary_writer.as_default():
            tf.summary.scalar('test-acc', float(total_correct/total), step=step)    #输出测试集的准确率
            tf.summary.image("val-onebyone-images:", val_images, max_outputs=64, step=step)     #输出36张图片
            
            val_images = tf.reshape(val_images, [-1, 28, 28])
            figure = image_grid(val_images)
            tf.summary.image('val-images:', plot_to_image(figure), step=step)