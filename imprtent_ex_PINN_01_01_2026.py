#  ODE: dy/dx + 3y = 0
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input

def boundery(x):
    return [np.sin(x)]      

def construct_training_set_BC(dx):
    x = np.pi*np.arange(-1, 1.1 , 0.1, dtype = np.float32)
    zeros = np.zeros(np.shape(x))
    y_train_BC = np.array(boundery(x))[0]
    x_train_BC = np.vstack((x, zeros)).T
    #print ('x_train_BC type = ',type(x_train_BC))
    #print ('y_train_BC type = ',type(y_train_BC))
    #print ('x_train_BC = ',x_train_BC)
    #print ('y_train_BC = ',y_train_BC)
    return x_train_BC,y_train_BC

def construct_training_set_IC(dt):
    t =  (np.arange(0, 1.0001 , dt, dtype = np.float32))
    Nt = np.shape(t)
    x1 = np.ones(Nt)
    xminus1 = x1-2.0
    x1_t = np.vstack((x1, t)).T.tolist()
    x_minus1_t = np.vstack((xminus1, t)).T.tolist()
    x_train_IC = np.array( x_minus1_t+x1_t)
    y_train_IC = np.zeros(np.shape(x_train_IC)[0])
    #print ('x_train_IC = ',x_train_IC)
    #print ('y_train_IC = ',y_train_IC)
    #print (np.shape(x_train_IC)[0])
    return x_train_IC,y_train_IC


def construct_training_set_sampled(N_points):
    xt = []
    for i in range(0, N_points):
      xt.append([np.random.uniform(-1,1),np.random.uniform(0,1)])
    xt = np.array(xt)
    y_IC = np.zeros(N_points)
    #print (xt)
    return xt,y_IC

def construct_training_set(dx,dt,N_points):
    x_train_BC, y_train_BC = construct_training_set_BC(dx)
    x_train_IC, y_train_IC = construct_training_set_IC(dt)    
    x_train_S,  y_train_S  = construct_training_set_sampled(N_points)
    x_train =  np.array(x_train_BC.tolist() + x_train_IC.tolist())
    y_train =  np.array(y_train_BC.tolist() + y_train_IC.tolist())
    #x_train[0][0]=1 # just for checking hessian calculation with fixed NN weihgts
    #x_train[0][1]=2
    #print ('x_train = ',x_train)    
    #print ('y_train = ',y_train)
    #print ('x_train_S = ',)
    return x_train,y_train, x_train_S,  y_train_S

def init_model1(num_hidden_layers=1, num_neurons_per_layer=2):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(2,))) 
    # Append hidden layers
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer,activation=tf.keras.activations.exponential,kernel_initializer='glorot_normal'))
    # Output is one-dimensional
    model.add(tf.keras.layers.Dense(1))
    #print (model.get_layer(index=1))
    return model

def init_model2():
    inputs = Input(shape=(2,))
    x = Dense(16, 'tanh')(inputs)
    x = Dense(8, 'tanh')(x)
    x = Dense(4)(x)
    y = Dense(1)(x)
    model = Model(inputs=inputs, outputs=y)
    return model

def set_weights_for_specific_model_layer(model):
    layer_weights_shape = [w.shape for w in model.get_weights()]
    #print(f"layers weight shapes: {layer_weights_shape}")
    custom_kernel = np.ones(layer_weights_shape[0]) 
    custom_bias = np.zeros(layer_weights_shape[1])
    model.set_weights([custom_kernel, custom_bias]) 
    
def set_weights_as_one_for_model(model):
    current_weights = model.get_weights()
    new_weights = [np.ones_like(w) for w in current_weights]
    model.set_weights(new_weights)  
    
def set_internal_variables_for_model(model,m,b):
    tm = tf.convert_to_tensor(m)
    tb = tf.convert_to_tensor(b)
    first_layer  = model.layers[0]
    secend_layer = model.layers[1]
    set_weights_for_specific_model_layer(first_layer)
    set_weights_for_specific_model_layer(secend_layer)
    dense_layer = model.get_layer(index=0)
    dense_layer.set_weights([tm,tb])
    #print (' network trainable_variables  = ')
    #print( model.trainable_variables)
    #print ('')
    #print (' W1  = ')
    #print( model.trainable_variables[0].numpy())
    #print (' b1 = ')
    #print( model.trainable_variables[1].numpy())
    #print (' W2 = ')
    #print( model.trainable_variables[2].numpy())
    #print (' b2 = ')
    #print( model.trainable_variables[3].numpy())

def excat_solution(model,r):
    w11=model.trainable_variables[0][0][0].numpy()
    w12=model.trainable_variables[0][0][1].numpy()
    w21=model.trainable_variables[0][1][0].numpy()
    w22=model.trainable_variables[0][1][1].numpy()
    w13=model.trainable_variables[2][0].numpy()
    w23=model.trainable_variables[2][1].numpy()
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


def one_epoch_fully_batched(model,opt,x_train, y_train,dataset_S,epoch_nnumber):    
   
    with tf.GradientTape() as model_tape:  
        l1_temp = loss_fn(y_train,model(x_train))   
        l1 = tf.cast(l1_temp, dtype=tf.float64)
        
        for step, (x,y_true) in enumerate(dataset_S):
            x_variable = tf.Variable(x) 
            with tf.GradientTape(persistent=True) as loss_tape2:
                loss_tape2.watch(x_variable) 
                with tf.GradientTape(persistent=True) as loss_tape1:
                    loss_tape1.watch(x_variable)
                    u = model(x_variable)
                    grad_u = loss_tape1.gradient(u,x_variable)        
                hessian_u = loss_tape2.jacobian(grad_u, x_variable)    
                ux = grad_u[0][0]
                ut = grad_u[0][1]
                uxx = hessian_u[0][0][0][0]
                u  = u.numpy()          
                l2_abs = tf.abs(ut + u*ux - (0.001)*uxx)
                if step==0:
                    l2 = l2_abs
                else:         
                    l2 = l2 + l2_abs
                #print (step, l2)
        loss = tf.math.reduce_mean(l1+l2)
        grad = model_tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grad, model.trainable_variables))
        #print ('l1 = ',l1)
        #print ('l2 = ',l2)
        print(f"epoch_nnumber {epoch_nnumber}: l1 = {l1} l2 = {l2} loss={loss.numpy()}")
    
    
    
    

N_points=30
sample_number = 4
N_epochs=20
loss_fn = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size',name='mean_squared_error')
x_train, y_train,x_train_S,  y_train_S = construct_training_set(0.1,0.1,N_points)
x_train = tf.convert_to_tensor(x_train)
y_train = tf.convert_to_tensor(y_train)
x_train_S = tf.convert_to_tensor(x_train_S)
y_train_S = tf.convert_to_tensor(y_train_S)
dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(1)
dataset_S = tf.data.Dataset.from_tensor_slices((x_train_S,y_train_S)).batch(1)
#sample_number = 4 m = [[1,2],[3,4]] b = [0,0] set_internal_variables_for_model(model,m,b)

model = init_model2()
model.summary()

#print (loss_fn(y_train[:sample_number],model(x_train[:sample_number]).numpy()))


opt = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99)
#one_epoch_fully_batched(model,opt,x_train, y_train,dataset_S,1)




for i in range(0,N_epochs):
    one_epoch_fully_batched(model,opt,x_train, y_train,dataset_S,i)




    
# for step, (x,y_true) in enumerate(dataset_S):
#     #l1 = loss_fn(y_train,model(x_train).numpy())   
#     #l1_c = tf.cast(l1, dtype=tf.float64)
#     x_variable = tf.Variable(x) 
#     with tf.GradientTape() as model_tape:
#         l1 = loss_fn(y_train,model(x_train))   
#         l1_c = tf.cast(l1, dtype=tf.float64)
#         with tf.GradientTape(persistent=True) as loss_tape2:
#             loss_tape2.watch(x_variable)  # "The input x must be a tf.Variable or explicitly "watched" using tape.watch(x) to be tracked by GradientTape "    
#             with tf.GradientTape(persistent=True) as loss_tape1:
#                 loss_tape1.watch(x_variable)
#                 u = model(x_variable)
#                 grad_u = loss_tape1.gradient(u,x_variable)        
#             hessian_u = loss_tape2.jacobian(grad_u, x_variable)    
#         ux = grad_u[0][0]
#         ut = grad_u[0][1]
#         uxx = hessian_u[0][0][0][0]
#         u  = u.numpy()
#         #print ('u',np.shape(u))  
#         #print ('gradient: ',grad_u)
#         #print ('hessian',hessian_u)
#         l2 = ut + u*ux - (0.001)*uxx
#         l2_abs = tf.abs(l2)
#         loss = tf.math.reduce_mean(l1_c+l2_abs)
#         #loss = tf.math.reduce_mean(l2_abs)
      

        
#     grad = model_tape.gradient(loss, model.trainable_variables)
#     #print ('step = ',step)
#     #print (grad)
#     #print ('')
#     opt.apply_gradients(zip(grad, model.trainable_variables))
#     if step%20==0:
#         print(f"Step {step}: loss={loss.numpy()}")



