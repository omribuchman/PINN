import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)
print (tf.config.list_physical_devices(device_type=None)) # https://www.tensorflow.org/api_docs/python/tf/config/list_physical_devices
print('tf.version = ',tf.__version__)
print ('tf.executing_eagerly = ',tf.executing_eagerly())

#    f:R^3 --> R 
#    layer = f(x) = relu(wx+b)
#    x --w11-->    y
#    t --w12-->    y

layer = tf.keras.layers.Dense(1, activation='relu',trainable = True)
xt = tf.Variable([[1., 2.]]) # set xt[0] = x, xt[1] = t
with tf.GradientTape(persistent=True) as tape: 
  u = layer(xt)
  loss = tf.reduce_mean(u**2)

print ('u = ',u.numpy())
print (' layer.trainable_variables = ', layer.trainable_variables)
#layer.trainable_variables[0].assign([[1],[1],[1]]) # fixe weights for now 
'Calculate gradients with respect to every trainable variable'
grad_w = tape.gradient(loss, layer.trainable_variables)
#print ('grad_w = ',grad_w)
grad_xt = tape.gradient(u, xt)
du_dx = grad_xt[0][0].numpy()
du_dt = grad_xt[0][1].numpy()
print ('du_dx = ',du_dx)
print ('du_dt = ',du_dt)
loss = du_dx-du_dt





x = tf.Variable([1.0, 2.0])

with tf.GradientTape(persistent=True) as tape2:
    with tf.GradientTape(persistent=True) as tape1:
        y = tf.reduce_sum(x * x) # Example function: y = x[0]^2 + x[1]^2
    
    gradient = tape1.gradient(y, x)
    # The tape.jacobian call computes the Hessian matrix
    hessian = tape2.jacobian(gradient, x)

print("Gradient:", gradient.numpy())
print("Hessian Matrix:\n", hessian.numpy())
