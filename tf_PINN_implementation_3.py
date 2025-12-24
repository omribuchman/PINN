import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
# NOTE - x_train[:i] - all samples in TS up to i , not the i'th sample !!
# u(x,t)
# Burgers eq.: u_t+uu_x-(0.001)u_xx = 0
# BC         : u(x,0) = sin(x)
# IC         : u(-1,t) = i(1,t) = 0 

def boundery(x):
    return [np.sin(x)]      

def construct_training_set_BC(dx):
    x = np.pi*np.arange(-1, 1.1 , 0.1, dtype = np.float32)
    zeros = np.zeros(np.shape(x))
    y_train = np.array(boundery(x))[0]
    x_train = np.vstack((x, zeros)).T
    #x_train =  tf.convert_to_tensor(np.vstack((x, zeros)).T)
    print (type(y_train))
    #print (x_train)
    #print ('y_train shape: ',np.shape(y_train))
    #print('x_train     = ',x_train)
    #print('x_train[:1] = ',x_train[:1])
    #print('x_train[:1] shape',np.shape(x_train[:1])) 





    return x_train,y_train

def construct_training_set_IC(dt):
    t =  (np.arange(0, 1.0001 , dt, dtype = np.float32))

    Nt = np.shape(t)
    x1 = np.ones(Nt)
    xminus1 = x1-2.0
    x1_t = np.vstack((x1, t)).T.tolist()
    x_minus1_t = np.vstack((xminus1, t)).T.tolist()
    x_train_IC = np.array( x_minus1_t+x1_t)
    y_train_IC = np.zeros(2*Nt)
    print ('x_train_IC type = ',type(x_train_IC))
    print ('y_train_IC type = ',type(y_train_IC))
    
    return x_train_IC,y_train_IC
            
    
    
    
    
    
  




def construct_training_set2(dx):
    x = np.pi*np.arange(-1, 1.1 , dx, dtype = np.float32)
    #print (np.shape(x))
    x_train = np.zeros((np.size(x), 1, 2))
    j=0
    for i in x:
        x_train[j][0][0] = i
        j = j+1
    #print (x_train)
    #print ('x_train shape = ',np.shape(x_train))
    y_train = np.array(boundery(x))
    #print ('y_train[0] ',y_train[0])
    #print ('y_train[0] shape ',np.shape(y_train[0]))
    #print('x_train     = ',x_train)
    #print('x_train[:1] = ',x_train[:1])
    #print('x_train[:1] shape',np.shape(x_train[:1]))
    return x_train, y_train[0]  

def init_model(num_hidden_layers=1, num_neurons_per_layer=2):
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

def loss_fn_1(model, x): # use to be get_r_simple2 
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)  # "The input x must be a tf.Variable or explicitly "watched" using tape.watch(x) to be tracked by GradientTape "    
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            u = model(x)
            u_x = tape1.gradient(u, x)        
        hessian = tape2.jacobian(u_x, x)         
    #print ('gradient: ',u_x.numpy())
    #print ('hessian:',hessian.numpy())
    du_dt = u_x[0][1]
    du_dx_dx = hessian[0][0][0][0]
    #print ('du_dt   : ',du_dt.numpy())
    #print ('du_dx_dx: ',du_dx_dx.numpy())
    loss = du_dt - du_dx_dx
    return  loss

#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size',name='mean_squared_error')

def custom_mse_loss(y_true, y_pred):
    l = loss_fn(y_true,y_pred)
    #squared_difference = tf.square(y_true - y_pred) 
    #return tf.reduce_mean(squared_difference, axis=-1)
    return l



#--------------------MAIN-------------------------- 
#x_train,y_train = construct_training_set2() # (21,1,2)
x_train,y_train = construct_training_set_BC(0.1)   # (1,2)
construct_training_set_IC(0.1)   # (1,2)


model = init_model()
#model.summary()

sample_number = 3
predictions = model(x_train[:sample_number]).numpy()
#print (predictions)
#print (y_train[:sample_number])

#print ('loss_fn = ',loss_fn(y_train[:sample_number],predictions).numpy())

#print ('custom_mse_loss = ',custom_mse_loss(y_train[:sample_number],predictions).numpy())




#model.compile(optimizer='adam', loss=custom_mse_loss)
#model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy']) #  loss=custom_mse_loss

#model.fit(x_train, y_train, epochs=5)






