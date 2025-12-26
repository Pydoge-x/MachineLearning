import tensorflow as tf

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# 检查GPU详情
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print(gpu)
    details = tf.config.experimental.get_device_details(gpu)
    print(details)
# 测试TensorFlow是否能使用GPU进行计算
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
c = tf.matmul(a, b, transpose_b=True)
print(c)