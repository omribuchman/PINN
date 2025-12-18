from typing import Dict
import tensorflow as tf
import numpy as np
import pinnstf2
tf.config.run_functions_eagerly(True)

# relevant file are in directoris: /home/omri/.local/lib/python3.11/site-packages/pinnstf2/data/mesh
# this is "Tutorial - 0: Continuous Forward Schrodinger". taken from: https://github.com/rezaakb/pinns-tf2/blob/main/tutorials/0-Schrodinger.ipynb
# started at 30.10.2025
# Basic structure:   u(x,t) = real part, v(x,t)=imagenary part, h=abs(u,v) 
# Initial condition: u(x,0) = sec(x),    v(x,0)=0 
# function_mapping in pinn_data_modulo.py file  
# loss_fn in mesh_sampler
# define sampling points in  function collection_points located at: mesh.py 
# NOTE: tf.config.run_functions_eagerly(True)




# Define Mesh
# ___________
# "PINNs require a discretized domain (mesh) over which the physical equations are solved

def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.
    :param root_path: The root directory containing the data.
    :return: Processed data will be used in Mesh class.
    """
    # read the exact solution from file /home/omri/pinnstf2/data/NLS.mat    
    data = pinnstf2.utils.load_data(root_path, "NLS.mat")
    exact = data["uu"]
    exact_u = np.real(exact) # (Nx=256,T=201)
    exact_v = np.imag(exact) # (Nx=256,T=201)
    #print (exact_u.shape)
    exact_h = np.sqrt(exact_u**2 + exact_v**2) # N x T
    return {"u": exact_u, "v": exact_v, "h": exact_h}

    """ Defining Time and Spatial Domains Separately: Here, we use pinnstf2.data.TimeDomain 
    and pinnstf2.data.Interval for creating 1-D spatial domains.
    These domains are then used to define a pinnstf2.data.Mesh."""

time_domain = pinnstf2.data.TimeDomain(t_interval=[0, 1.57079633], t_points = 201)
spatial_domain = pinnstf2.data.Interval(x_interval= [-5, 4.9609375], shape = [256, 1])
mesh = pinnstf2.data.Mesh(root_dir='data',read_data_fn=read_data_fn,spatial_domain = spatial_domain,time_domain = time_domain)

"""Define Train datasets
    For solving Schrodinger PDE, we have:
    1. Initial condition u(x,0) = sec(x), v(x,0)=0 by either:
        i.  Sample from the data
        ii. Defining a function for calculating initial condition
    2. Periodic boundary condition
    3. Collection points for the PDE."""
    
# sample IC from data:
N0 = 50 # number of sample points for IC. 
in_c = pinnstf2.data.InitialCondition(mesh = mesh,num_sample = N0,solution = ['u', 'v'])
#print (n_c.solution['u'][0]) # how to acces dict elements solution
#print (n_c.solution_sampled[0])   
# n_c.solution = is all the points(256) of the IC, while n_c.solution_sampled is 50 random samples of them (note that diffrent values are shown for diffrent operations)

# sample Boundary Condition:
# Periodic Boundary Condition: note: at diffrent t's, u(0,t) =f(t) !! so we have to smaple it !
N_b = 50
pe_b = pinnstf2.data.PeriodicBoundaryCondition(mesh = mesh,
                                                 num_sample = 50,
                                                 derivative_order = 1,
                                                 solution = ['u', 'v'])

N_f = 1000
me_s = pinnstf2.data.MeshSampler(mesh = mesh,num_sample = N_f,collection_points = ['f_v', 'f_u'])
#print ('me_s',me_s.spatial_domain_sampled) # at this point only the sampled points had been chosen. 

#Define Validation dataset
val_s = pinnstf2.data.MeshSampler(mesh = mesh,solution = ['u', 'v', 'h'])
#print (val_s.spatial_domain_sampled)

# Define Neural Networks
# "...the inputs of this network are x and t, and the outputs of this network are u and v"
net = pinnstf2.models.FCN(layers = [2, 100, 100, 100, 100, 2],output_names = ['u', 'v'],lb=mesh.lb,ub=mesh.ub)

# Define pde_fn and output_fn functions
# output_fn function:
# Outputs: It is output of the network. In our case, this dictionary should have two output: u and v.    
def output_fn(outputs: Dict[str, tf.Tensor],
              x: tf.Tensor,
              t: tf.Tensor):
    """Define `output_fn` function that will be applied to outputs of net."""

    outputs["h"] = tf.sqrt(outputs["u"] ** 2 + outputs["v"] ** 2)

    return outputs
    
def pde_fn(outputs: Dict[str, tf.Tensor],
           x: tf.Tensor,
           t: tf.Tensor):   
    """Define the partial differential equations (PDEs)."""
    
    u_x, u_t = pinnstf2.utils.gradient(outputs["u"], [x, t])
    v_x, v_t = pinnstf2.utils.gradient(outputs["v"], [x, t])

    u_xx = pinnstf2.utils.gradient(u_x, x)[0]
    v_xx = pinnstf2.utils.gradient(v_x, x)[0]

    outputs["f_u"] = u_t + 0.5 * v_xx + (outputs["u"] ** 2 + outputs["v"] ** 2) * outputs["v"]
    outputs["f_v"] = v_t - 0.5 * u_xx - (outputs["u"] ** 2 + outputs["v"] ** 2) * outputs["u"]

    return outputs


# Define PINNDataModule
train_datasets = [me_s, in_c, pe_b]
val_dataset = val_s
datamodule = pinnstf2.data.PINNDataModule(train_datasets = [me_s, in_c, pe_b],
                                            val_dataset = val_dataset,
                                            pred_dataset = val_s)

# Define PINNModule
model = pinnstf2.models.PINNModule(net = net,
                                   pde_fn = pde_fn,
                                   output_fn = output_fn,
                                   loss_fn = 'mse')

trainer = pinnstf2.Trainer(max_epochs=20000, check_val_every_n_epoch=1000)

trainer.fit(model=model, datamodule=datamodule)
# trainer.validate(model=model, datamodule=datamodule)
# preds_dict = trainer.predict(model=model, datamodule=datamodule)
# pinnstf2.utils.plot_schrodinger(mesh=mesh,
#                                  preds=preds_dict,
#                                  train_datasets=train_datasets,
#                                  val_dataset=val_dataset,
#                                  file_name='out')
















