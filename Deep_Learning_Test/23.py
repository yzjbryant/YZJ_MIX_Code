import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
# sess=tf.Session()
# a=tf.constant(1)
# b=tf.constant(3)
# print(sess.run(a+b))
# print(tf.__version__)

x=tf.Variable(3,name="x")
y=tf.Variable(4,name="y")
f=x*x*y+y+2+y*y*y*y*y*y*y*y*y
# sess=tf.Session()
# sess.run(x.initializer)
# sess.run(y.initializer)
# result=sess.run(f)
# print(result)
# sess.close()

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result=f.eval()
print(result)


c=tf.contrib.learn()
d=tf.contrib.slim()
e=tf.keras()
import tensorboard
