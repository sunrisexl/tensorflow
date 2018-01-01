'''
tensorflow学习
识别手写体
1.定义算法公式，也就是神经网络forward时的计算
2.定义loss，选定优化器，指定优化器优化loss
3.迭代地对数据进行训练，
4.在测试集上对准确率进行评测
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 加载mnist数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)


sess = tf.InteractiveSession()  # 交互式进程
x = tf.placeholder(tf.float32, [None, 784])  # 输入数据

w = tf.Variable(tf.zeros([784, 10]))  # variable用来存储模型参数
b = tf.Variable(tf.zeros([10]))  # weights和biases初始化为0

y = tf.nn.softmax(tf.matmul(x, w) + b)  # 实现saftmax Regression算法

# 定义损失函数loss function 来描述模型对问题的分类精度
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 使用随机梯度下降SGD优化算法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 全局参数初始化器，执行其run方法
tf.global_variables_initializer().run()

# 迭代执行训练操作
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
