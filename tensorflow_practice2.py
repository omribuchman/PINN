import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.config.run_functions_eagerly(True)
print(tf.__version__)

x = tf.Variable(tf.zeros(shape=[5], dtype=tf.float32))
x[2].assign(1.2) # Correct way to modify a tf.Variable

with tf.Session() as sess:
     result = sess.run(x)
     print (result)

# #print (x.get_shape())

# #print (tf.constant([[1,2],[3,4],[5,6]], dtype=tf.float32, name="my_tensor"))

#A = pd.DataFrame(np.random.normal(0,1,10)) 
#B = pd.DataFrame(np.random.normal(0,1,10)) 



#print (tf.constant([A,B]), dtype=tf.float32, name="my_tensor")





# # Scalar tensor (0-dimensional)
# scalar_tensor = tf.constant(5)
# print(f"Scalar Tensor: {scalar_tensor}")

# # Vector tensor (1-dimensional)
# vector_tensor = tf.constant([1, 2, 3, 4])
# print(f"Vector Tensor: {vector_tensor}")

# # Matrix tensor (2-dimensional)
# matrix_tensor = tf.constant([[1, 2], [3, 4]])
# print(f"Matrix Tensor: {matrix_tensor}")

# # 3D tensor
# tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# print(f"3D Tensor: {tensor_3d}")




# #Create a variable tensor
# variable_tensor = tf.Variable(np.random.normal(0, 1.0,10))
# print(f"Initial Variable Tensor: {variable_tensor}")

# # Modify the variable tensor
# #variable_tensor.assign([4, 5, 6])
# #print(f"Modified Variable Tensor: {variable_tensor}")



# #standard_normal_vector = np.random.normal(size=10)
# #print(np.shape(  standard_normal_vector))

# from pyDOE import lhs


# A = np.random.normal(0,1,3) 
# B = np.random.normal(0,1,3)

# print ('matrix_hstack')
# matrix_hstack = np.hstack((A.reshape(-1, 1), B.reshape(-1, 1)))
# print(matrix_hstack)
# print ('')
# print ('10 * lhs(2, 3)')
# f =  10 * lhs(2, 3)
# print (f)
# print ('')
# print (matrix_hstack-f)




# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior() # To use TensorFlow 1.x behavior

# # Define a simple computational graph
# a = tf.constant(5)
# b = tf.constant(3)
# c = tf.add(a, b)
# a.numpy()
# # Create and run a session
# #with tf.Session() as sess:
# #     result = sess.run(a)
# #     print(result) # Output: 8
# #     print (type(result))   
# #     #print (type(c))   
    

