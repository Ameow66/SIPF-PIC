import jax
import jax.numpy as jnp
from jax import random,  vmap, jit
from jax import lax
from jax import config
from functools import partial
import time
import numpy as np
import matplotlib.pyplot as plt

def gen_initX_ball(key,J,r,pos):
    k1,k2,k3=random.split(key,3)
    #
    X=random.normal(k1,(J,3))
    X=X/jnp.linalg.norm(X,axis=1,keepdims=True)
    X=X*(random.uniform(k2,(J,1))**(1/3))*r
    X=X+random.choice(k3,pos,shape=(J,))
    '''Test below'''
    plt.hist2d(X[:,0],X[:,1])
    return X

def gen_initX(key, J, r, pos, plot=False):
    num_spheres = len(pos)
    
    J_adjusted = ((J + num_spheres - 1) // num_spheres) * num_spheres
    points_per_sphere = J_adjusted // num_spheres
    
    k1, k2, k3 = random.split(key, 3)
    
    X = random.normal(k1, (J_adjusted, 3))
    X = X / jnp.linalg.norm(X, axis=1, keepdims=True)
    X = X * (random.uniform(k2, (J_adjusted, 1)) ** (1/3)) * r

    offsets = jnp.repeat(pos, points_per_sphere, axis=0) 
    X = X + offsets
    

    return X[:J] if J != J_adjusted else X 