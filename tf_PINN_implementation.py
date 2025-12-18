# taken from https://georgemilosh.github.io/blog/2022/distill/
# start to work at 10.12.2025
import tensorflow as tf
import numpy as np
  
# Set data type
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)


# Define residual of the PDE

def init_model_simple(num_hidden_layers=1, num_neurons_per_layer=2):
    # Initialize a feedforward neural network
    model = tf.keras.Sequential()
    # Input is one-dimensional (time + one spatial dimension)
    #model.add(tf.keras.Input(1)) # this is the original code but it is not working !
    model.add(tf.keras.Input(shape=(1,))) 


    # Append hidden layers
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
            activation=tf.keras.activations.exponential,
            kernel_initializer='glorot_normal'))
    # Output is one-dimensional
    model.add(tf.keras.layers.Dense(1))
    return model

model_simple = init_model_simple()


def get_r_simple(model, x):
    # A tf.GradientTape is used to compute derivatives in TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        # Determine residual 
        u = model(x)
        # Compute gradient u_x within the GradientTape
        # since we need second derivatives
        u_x = tape.gradient(u, x)
    del tape
    return u_x


w21 = model_simple.weights[2].numpy()[0] # last layer
w22 = model_simple.weights[2].numpy()[1]

print (w21,w22)

w11 = model_simple.weights[0].numpy()[0,0] # first layer
w12 = model_simple.weights[0].numpy()[0,1]

print (w11,w12)

x = tf.constant([1], dtype=tf.float32)

print(f'du/dx = {get_r_simple(model_simple, tf.constant(x, dtype=tf.float32)).numpy()} = {w21*w11*np.exp(w11*x) + w22*w12*np.exp(w12*x)}' )


def compute_loss_simple(model, X_r):
    return get_r_simple(model, X_r)

def get_grad_simple(model, X_r):
    
    with tf.GradientTape(persistent=True) as tape:
        # This tape is for derivatives with
        # respect to trainable variables
        #tape.watch(model.trainable_variables)
        loss = compute_loss_simple(model, X_r)

    g = tape.gradient(loss, model.trainable_variables)
    del tape

    return loss, g

loss, g = get_grad_simple(model_simple, x)
for gi, varsi in zip(g, model_simple.variables):
    print(f'{varsi.name} has graidents {gi}')



print(f"The value we expect for dl/dw11 = {w21*(1+w11**2)*np.exp(w11*x)}, dl/dw12 = {w22*(1+w12*2)*np.exp(w12*x)}")







