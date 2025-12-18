# taken from https://georgemilosh.github.io/blog/2022/distill/
# start to work at 10.12.2025
import tensorflow as tf
import numpy as np
  
# Set data type
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)


# Define residual of the PDE

def init_model_simple(num_hidden_layers=1, num_neurons_per_layer=2):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(2,))) 
    # Append hidden layers
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
            activation=tf.keras.activations.exponential,
            kernel_initializer='glorot_normal'))
    # Output is one-dimensional
    model.add(tf.keras.layers.Dense(1))
    #print (model.get_layer(index=1))
    return model

model_simple = init_model_simple()
print (model_simple.get_layer(index=1))


def set_weights_for_specific_model_layer(model):
    layer_weights_shape = [w.shape for w in model.get_weights()]
    #print(f"layers weight shapes: {layer_weights_shape}")
    custom_kernel = np.ones(layer_weights_shape[0]) 
    custom_bias = np.zeros(layer_weights_shape[1])
    model.set_weights([custom_kernel, custom_bias]) 
    


def set_weights_as_one_for_model(model):
    current_weights = model_simple.get_weights()
    new_weights = [np.ones_like(w) for w in current_weights]
    model_simple.set_weights(new_weights)  


m = [[1,2],[3,4]]
b = [0,0]
tm = tf.convert_to_tensor(m)
tb = tf.convert_to_tensor(b)



first_layer  = model_simple.layers[0]
secend_layer = model_simple.layers[1]
set_weights_for_specific_model_layer(first_layer)
set_weights_for_specific_model_layer(secend_layer)
dense_layer = model_simple.get_layer(index=0)
dense_layer.set_weights([tm,tb])

print (' network trainable_variables  = ')
print( model_simple.trainable_variables)
print ('')
print (' W1  = ')
print( model_simple.trainable_variables[0].numpy())
print (' b1 = ')
print( model_simple.trainable_variables[1].numpy())
print (' W2 = ')
print( model_simple.trainable_variables[2].numpy())
print (' b2 = ')
print( model_simple.trainable_variables[3].numpy())

def get_r_simple(model, x):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)  # "The input x must be a tf.Variable or explicitly "watched" using tape.watch(x) to be tracked by GradientTape "    
        u = model(x)
        u_x = tape.gradient(u, x)
        hessian = tape.jacobian(u_x, x)
        print ('gradient =')
        print (u_x.numpy())
        print ('estimated hessian =')
        print (hessian.numpy())
        #u_x_x = tape.gradient(u_x, x)
        #print ('u_x_x = ')
        #print (u_x_x.numpy())
        del tape
    return u_x,hessian

def get_r_simple2(model, x):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)  # "The input x must be a tf.Variable or explicitly "watched" using tape.watch(x) to be tracked by GradientTape "    
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            u = model(x)
            u_x = tape1.gradient(u, x)
        #print (u_x)
        d2y_dx2 = tape2.gradient(u_x, x) 
        hessian = tape2.jacobian(u_x, x)         
    #print('dy_dx:  ', u_x.numpy())   # Expected: 9.0 (3 * 3 * x**2)
    #print('d2y_dx2:', d2y_dx2.numpy())
    print ('hessian:',hessian.numpy())
    return u_x,hessian

def excat_solution(model_simple,r):
    w11=model_simple.trainable_variables[0][0][0].numpy()
    w12=model_simple.trainable_variables[0][0][1].numpy()
    w21=model_simple.trainable_variables[0][1][0].numpy()
    w22=model_simple.trainable_variables[0][1][1].numpy()
    w13=model_simple.trainable_variables[2][0].numpy()
    w23=model_simple.trainable_variables[2][1].numpy()
    x = r.numpy()[0][0]
    y = r.numpy()[0][1]
    # print ('w11 w12 = ',w11,w12)
    # print ('w21,w22 = ',w21,w22)    
    # print ('w13,w23 = ',w13,w23)
    # print ('x, y = ',x,y)
    du_dx_dx = w13*w11**2*np.exp(w11*x + w21*y) + w23*w12**2*np.exp(w12*x + w22*y) 
    du_dy_dy = w13*w21**2*np.exp(w11*x + w21*y) + w23*w22**2*np.exp(w12*x + w22*y)    
    print ('exact du_dx_dx, du_dy_dy = ',du_dx_dx,du_dy_dy)
    return (du_dx_dx,du_dy_dy)


x = tf.constant([[1,2]], dtype=tf.float32)
#ux,hessian = get_r_simple(model_simple, tf.constant(x, dtype=tf.float32))
ux,hessian = get_r_simple2(model_simple, tf.constant(x, dtype=tf.float32))
excat_solution(model_simple,x)







# def compute_loss_simple(model, X_r):
#     return get_r_simple(model, X_r)

# def get_grad_simple(model, X_r):
    
#     with tf.GradientTape(persistent=True) as tape:
#         # This tape is for derivatives with
#         # respect to trainable variables
#         #tape.watch(model.trainable_variables)
#         loss = compute_loss_simple(model, X_r)

#     g = tape.gradient(loss, model.trainable_variables)
#     del tape

#     return loss, g

# loss, g = get_grad_simple(model_simple, x)
# for gi, varsi in zip(g, model_simple.variables):
#     print(f'{varsi.name} has graidents {gi}')






# def compute_loss(model, x):
#     u = model(x)
#     return u




# def get_r_simple_2(model, x):
#     with tf.GradientTape() as t2:
#         with tf.GradientTape() as t1:
#             loss = compute_loss(x)    
#             # Calculate the first-order gradients
#             # This result is a list of tensors, one for each variable
#             grads = t1.gradient(loss, trainable_variables)
    
#     # Calculate the second-order gradients (Hessian) by computing the jacobian of the first-order gradients
#     # t2 needs to watch the variables again for the second derivative calculation
#     # The result is a list of matrices/tensors
#     hessian = t2.jacobian(grads, trainable_variables)
