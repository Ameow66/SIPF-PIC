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

def gen_initX(key,J,r,pos,plot=False):
    k1,k2,k3=random.split(key,3)
    #
    X=random.normal(k1,(J,3))
    X=X/jnp.linalg.norm(X,axis=1,keepdims=True)
    X=X*(random.uniform(k2,(J,1))**(1/3))*r
    X=X+random.choice(k3,pos,shape=(J,))
    if plot:
        L=5
        plt3d_alpha=0.3
        plt3d_range=np.array([-L/4,L/4])
        fig=plt.figure(figsize=(6,2))
        ax1=fig.add_subplot(131)
        c1=ax1.hist2d(X[:,0],X[:,1],bins=40,range=[[-L/2,L/2],[-L/2,L/2]])[3]
        plt.colorbar(c1)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        ax2=fig.add_subplot(132)
        c2=ax2.hist2d(X[:,0],X[:,3],bins=40,range=[[-L/2,L/2],[-L/2,L/2]])[3]
        plt.colorbar(c2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('z')
        
        ax3=fig.add_subplot(133,projection='3d')
        ax3.plot3D(X[:,0],X[:,1],X[:,2],',', alpha=plt3d_alpha)
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('z')
        
        ax3.set_xlim(plt3d_range)
        ax3.set_ylim(plt3d_range)
        ax3.set_zlim(plt3d_range)
        
        plt.show()
        
    return X