import jax
import jax.numpy as jnp
from jax import random,  vmap, jit
from jax import lax
from jax import config
from functools import partial
import time
import numpy as np
import matplotlib.pyplot as plt



def gen_initX(key,J,r,pos,plot=False):
    k1,k2,k3=random.split(key,3)
    #
    X=random.normal(k1,(J,2))
    X=X/jnp.linalg.norm(X,axis=1,keepdims=True)
    X=X*(random.uniform(k2,(J,1))**(1/2))*r
    X=X+random.choice(k3,pos,shape=(J,))
    
        
    return X