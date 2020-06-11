import tensorflow as tf
a=tf.constant(1)
b=tf.constant(2)
sess=tf.Session()
print(sess.run(a+b))