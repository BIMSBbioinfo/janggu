import tensorflow as tf

def nan_to_zero(x):
  """ Replaces NaNs with zeros """
  return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

@tf.function
def random_orientation(x):
    #if tf.experimental.random.randint(2, size=1):
    if tf.random.uniform(minval=0, maxval=1, shape=(), dtype=tf.int32)>0:
        return x[::-1, ::-1, :]
    return x


@tf.function
def random_shift(data, shift):
    data = tf.cast(data, tf.float32)
    rshift = tf.random.uniform(minval=0, maxval=2*shift+1, shape=(), dtype=tf.int32)
    fil = tf.reshape(tf.one_hot(rshift, shift*2+1), [-1,1,1,1])
    fil = tf.repeat(fil, repeats=tf.shape(data)[-1], axis=2)
    fil = tf.repeat(fil, repeats=tf.shape(data)[-1], axis=3)
    fil = fil * tf.expand_dims(tf.eye(tf.shape(data)[-1]), 0)

    if len(tf.shape(data)) == 3:
        data = tf.expand_dims(data, 0)
    return tf.nn.convolution(data, fil, padding="SAME")[0]
