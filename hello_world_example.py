import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)
print (tf.config.list_physical_devices(device_type=None)) # https://www.tensorflow.org/api_docs/python/tf/config/list_physical_devices
print('tf.version = ',tf.__version__)
print ('tf.executing_eagerly = ',tf.executing_eagerly())

x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.]])
#print(x.numpy(),x.shape,x.dtype)


x = tf.Variable(3.0)
with tf.GradientTape() as tape:
  y = x**2
dy_dx = tape.gradient(y, x)
#print (dy_dx.numpy())


w = tf.Variable(tf.random.normal((3, 2)), name='w')
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
x = [[1., 2., 3.]]




# https://www.tensorflow.org/guide/autodiff
with tf.GradientTape(persistent=True) as tape:
  y = x @ w + b
  loss = tf.reduce_mean(y**2)
[dl_dw, dl_db] = tape.gradient(loss, [w, b])
#print(w)
#print (dl_dw.numpy())
#print (dl_db.numpy())

my_vars = {
    'w': w,
    'b': b
}
grad = tape.gradient(loss, my_vars)
#print (grad['b'])
#print (my_vars['b'].numpy())



layer = tf.keras.layers.Dense(1, activation='relu',trainable = False)
#    x1 --w11-->    y1     x1 --w21-->    y2  x1 --w31-->    y3
#    x2 --w12-->    y1     x2 --w22-->    y2  x2 --w31-->    y3
#    x3 --w13-->    y1     x3 --w23-->    y2  x3 --w31-->    y3
#    f:R^3 --> R^2 
#    layer = f(x) = relu(wx+b)
x = tf.Variable([[1., 2., 3. ]])
with tf.GradientTape(persistent=True) as tape:
  
  # Forward pass
  y = layer(x)
  #print (' layer.trainable_variables = ', layer.trainable_variables[0].numpy())

  loss = tf.reduce_mean(y**2)

print ('y = ',y.numpy())



#print (' layer.trainable_variables = ', layer.trainable_variables[0].numpy())
#layer.trainable_variables[0].assign([[1],[1],[1]]) # fixe weights for now 
#print (' layer.trainable_variables = ', layer.trainable_variables[0].numpy())
#y = layer(x)
#print('y = ',y.numpy())
#grad1 = tape.gradient(y, x)
#print ('grad1 = ',grad1)



#print('y**2 = ',(y**2).numpy())
#print ('loss = ',loss.numpy())
#layer.trainable_variables[0].assign([[1,1],[1,1],[1,1]])

#print (' layer.trainable_variables = ', layer.trainable_variables[1])
#y = layer(x)
#print('y = ',y[0][1].numpy())


# Calculate gradients with respect to every trainable variable
#grad = tape.gradient(loss, layer.trainable_variables)
#grad = tape.gradient(y, layer.trainable_variables)
#print (grad)


#print (grad[1])
# Calculate gradients with respect to input variable
#grad1 = tape.gradient(y, x)
#print ('grad1 = ',grad1)


#print ('type(grad) = ',type(grad)) 
#print ('grad = ',grad[1].numpy()) # grad = [df\fw,df\db]

#for var, g in zip(layer.trainable_variables, grad):
#  print(f'{var.name}, shape: {g.shape}')
#  print (var)



# # # A trainable variable
# x = tf.Variable(1.0, name='x')
# t = tf.Variable(1.0, name='t')
# w = tf.Variable(2.0, name='wheight')

# with tf.GradientTape() as tape:
#    u = (x**2) + (t**3) + (w**2) # u = u(x,t;w)

# grad = tape.gradient(u, [x, t, w])
# print (grad)

# for g in grad:
#    print('der = ',g.numpy())
#    #print (g)




# tf Graph Input
#X = tf.Variable("float")
#Y = tf.placeholder("float")
#print ('X = ',X)

# Set model weights
#W = tf.Variable(tf.random.normal(), name="weight")
#b = tf.Variable(tf.random.normal(), name="bias")

# Construct a linear model
#pred = tf.add(tf.mul(X, W), b)

# Mean squared error
#cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)




# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, Flatten
# from tensorflow.keras.models import Model

# # Define a symbolic input tensor
# input_tensor = Input(shape=(28, 28, 1), name='image_input')

# print (input_tensor)


# Build the model using symbolic tensors
#x = Flatten()(input_tensor)
#x = Dense(128, activation='relu')(x)
#output_tensor = Dense(10, activation='softmax')(x)

# Create a Keras Model from the symbolic inputs and outputs
#model = Model(inputs=input_tensor, outputs=output_tensor)

# The 'input_tensor' and 'output_tensor' here are symbolic tensors.
# They define the structure of the data expected by the model.




 