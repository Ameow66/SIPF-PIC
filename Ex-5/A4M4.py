import jax
import jax.numpy as jnp
from jax import random,  vmap, jit
from jax import config
from jax import lax
from functools import partial
import time
import numpy as np
import matplotlib.pyplot as plt
import queue

class PKS:
    def __init__(self,M0=100,H=24,L=8,k=0.1,eps=1e-3,reg=0,chi=1,mu=1,rng_key=random.PRNGKey(216)):
        start_time = time.time()
        self.H=2*int(H/2)
        self.L=L
        self.k=k
        self.eps=eps
        self.reg=reg
        self.chi=chi
        self.mu=mu
        self.key=rng_key
        self.M0=M0
        self.mlen = self.L / self.H
        self.Cv=jnp.zeros((self.H,self.H,self.H))
        self.varlog=[]
        self.tlog=[]
        self.cmaxlog=[]
        self.key, subkey = random.split(self.key)
        print("--- Build time %s s ---" % (time.time() - start_time))
        return

    def solve(self,t,t0,show_num=5,show_flag=True):

        self.P = self.X.shape[0]
        self.M = self.M0 / self.P
        self.offset = self.H / 2
        start_time0 = time.time()
        nt = int(t/t0)
        self.b2 = self.k**2+self.eps/t0
        u1 = jnp.arange(self.H).reshape(self.H, 1, 1)
        u2 = jnp.arange(self.H).reshape(1, self.H, 1)
        u3 = jnp.arange(self.H).reshape(1, 1, self.H)
        self.FK = - 1 / ((4*(jnp.pi**2)/(self.L**2)) * ((u1-self.offset)**2+(u2-self.offset)**2+(u3-self.offset)**2) + self.b2)
        coords = jnp.indices((self.H, self.H, self.H)).transpose(1, 2, 3, 0)
        self.FN = (coords - self.offset) * (2j * jnp.pi / self.L)
        start_time = time.time()
        
        for it in range(nt):
            self.key, subkey = random.split(self.key)
            self.X1 = self.Xstep(self.X, self.Cv, t0, subkey)



            self.Cv = self.Cstep(self.X, self.Cv, t0)

            
            self.X = self.X1
            self.varlog.append(jnp.var(self.X))
            self.tlog.append((it+1)*t0)
            self.cmaxlog.append(jnp.max(jnp.abs(jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.fftshift(self.Cv))))))


        
        print("--- Total time %s s ---" % (time.time() - start_time0))
        return

    
    @partial(jit, static_argnums=(0,))
    def Cstep(self, X, Cv, t0):
        # Compute positions
        u_pos = jnp.clip((X + self.L / 2) / self.mlen, 2 + 1e-8, self.H - 2 - 1e-8)
        u_int = jnp.floor(u_pos).astype(jnp.int32)
        u_frac = u_pos - u_int
    
        # Prepare for contributions
        u1, u2, u3 = u_int[:, 0], u_int[:, 1], u_int[:, 2]
        v1, v2, v3 = u_frac[:, 0], u_frac[:, 1], u_frac[:, 2]
    
        # Initialize rho
        rho = jnp.zeros((self.H, self.H, self.H))
    
        # Compute contributions using broadcasting
        rho = rho.at[u1, u2, u3].add((1 - v1) * (1 - v2) * (1 - v3) * (1 + v1/2 + v2/2 + v3/2 - v1*v1/2 - v2*v2/2 - v3*v3/2))
        rho = rho.at[u1 + 1, u2, u3].add(v1 * (1 - v2) * (1 - v3) * (1 + v1/2 + v2/2 + v3/2 - v1*v1/2 - v2*v2/2 - v3*v3/2))
        rho = rho.at[u1, u2 + 1, u3].add((1 - v1) * v2 * (1 - v3) * (1 + v1/2 + v2/2 + v3/2 - v1*v1/2 - v2*v2/2 - v3*v3/2))
        rho = rho.at[u1, u2, u3 + 1].add((1 - v1) * (1 - v2) * v3 * (1 + v1/2 + v2/2 + v3/2 - v1*v1/2 - v2*v2/2 - v3*v3/2))
        rho = rho.at[u1 + 1, u2 + 1, u3].add(v1 * v2 * (1 - v3) * (1 + v1/2 + v2/2 + v3/2 - v1*v1/2 - v2*v2/2 - v3*v3/2))
        rho = rho.at[u1 + 1, u2, u3 + 1].add(v1 * (1 - v2) * v3 * (1 + v1/2 + v2/2 + v3/2 - v1*v1/2 - v2*v2/2 - v3*v3/2))
        rho = rho.at[u1, u2 + 1, u3 + 1].add((1 - v1) * v2 * v3 * (1 + v1/2 + v2/2 + v3/2 - v1*v1/2 - v2*v2/2 - v3*v3/2))
        rho = rho.at[u1 + 1, u2 + 1, u3 + 1].add(v1 * v2 * v3 * (1 + v1/2 + v2/2 + v3/2 - v1*v1/2 - v2*v2/2 - v3*v3/2))
        
        rho = rho.at[u1 - 1, u2, u3].add(-(1 - v1) * (1 - v2) * (1 - v3) * v1 * (2 - v1) / 6)
        rho = rho.at[u1 + 2, u2, u3].add(-(1 - v1) * (1 - v2) * (1 - v3) * v1 * (1 + v1) / 6)
        rho = rho.at[u1 - 1, u2 + 1, u3].add(-(1 - v1) * v2 * (1 - v3) * v1 * (2 - v1) / 6)
        rho = rho.at[u1 + 2, u2 + 1, u3].add(-(1 - v1) * v2 * (1 - v3) * v1 * (1 + v1) / 6)
        rho = rho.at[u1 - 1, u2, u3 + 1].add(-(1 - v1) * (1 - v2) * v3 * v1 * (2 - v1) / 6)
        rho = rho.at[u1 + 2, u2, u3 + 1].add(-(1 - v1) * (1 - v2) * v3 * v1 * (1 + v1) / 6)
        rho = rho.at[u1 - 1, u2 + 1, u3 + 1].add(-(1 - v1) * v2 * v3 * v1 * (2 - v1) / 6)
        rho = rho.at[u1 + 2, u2 + 1, u3 + 1].add(-(1 - v1) * v2 * v3 * v1 * (1 + v1) / 6)

        rho = rho.at[u1, u2 - 1, u3].add(-(1 - v1) * (1 - v2) * (1 - v3) * v2 * (2 - v2) / 6)
        rho = rho.at[u1, u2 + 2, u3].add(-(1 - v1) * (1 - v2) * (1 - v3) * v2 * (1 + v2) / 6)
        rho = rho.at[u1 + 1, u2 - 1, u3].add(-v1 * (1 - v2) * (1 - v3) * v2 * (2 - v2) / 6)
        rho = rho.at[u1 + 1, u2 + 2, u3].add(-v1 * (1 - v2) * (1 - v3) * v2 * (1 + v2) / 6)        
        rho = rho.at[u1, u2 - 1, u3 + 1].add(-(1 - v1) * (1 - v2) * v3 * v2 * (2 - v2) / 6)
        rho = rho.at[u1, u2 + 2, u3 + 1].add(-(1 - v1) * (1 - v2) * v3 * v2 * (1 + v2) / 6)
        rho = rho.at[u1 + 1, u2 - 1, u3 + 1].add(-v1 * (1 - v2) * v3 * v2 * (2 - v2) / 6)
        rho = rho.at[u1 + 1, u2 + 2, u3 + 1].add(-v1 * (1 - v2) * v3 * v2 * (1 + v2) / 6)

        rho = rho.at[u1, u2, u3 - 1].add(-(1 - v1) * (1 - v2) * (1 - v3) * v3 * (2 - v3) / 6)
        rho = rho.at[u1, u2, u3 + 2].add(-(1 - v1) * (1 - v2) * (1 - v3) * v3 * (1 + v3) / 6)
        rho = rho.at[u1 + 1, u2, u3 - 1].add(-v1 * (1 - v2) * (1 - v3) * v3 * (2 - v3) / 6)
        rho = rho.at[u1 + 1, u2, u3 + 2].add(-v1 * (1 - v2) * (1 - v3) * v3 * (1 + v3) / 6)        
        rho = rho.at[u1, u2 + 1, u3 - 1].add(-(1 - v1) * v2 * (1 - v3) * v3 * (2 - v3) / 6)
        rho = rho.at[u1, u2 + 1, u3 + 2].add(-(1 - v1) * v2 * (1 - v3) * v3 * (1 + v3) / 6)
        rho = rho.at[u1 + 1, u2 + 1, u3 - 1].add(-v1 * v2 * (1 - v3) * v3 * (2 - v3) / 6)
        rho = rho.at[u1 + 1, u2 + 1, u3 + 2].add(-v1 * v2 * (1 - v3) * v3 * (1 + v3) / 6)
    
        # Perform FFT
        rhov = jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.fftshift(self.M * rho)))
        
        # Compute the convective term
        conv_term = -rhov - (self.eps / t0) * Cv
        
        return self.FK * conv_term

    @partial(jit, static_argnums=(0,))
    def Xstep(self, X, Cv, t0, key):
        Fgrad = self.FN * jnp.expand_dims(Cv, axis=-1)
        
        # Compute gradients
        grad0 = jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.fftshift(Fgrad, axes=(0, 1, 2)), axes=(0, 1, 2)), axes=(0, 1, 2))) / (self.mlen**3)

    
        # Compute positions and contributions
        u_pos = jnp.clip((X + self.L / 2) / self.mlen, 2 + 1e-8, self.H - 2 - 1e-8)
        u_int = jnp.floor(u_pos).astype(jnp.int32)
        u_frac = u_pos - u_int
    
        u1, u2, u3 = u_int[:, 0], u_int[:, 1], u_int[:, 2]
        v1, v2, v3 = u_frac[:, 0], u_frac[:, 1], u_frac[:, 2]

        v1 = jnp.expand_dims(v1, axis=-1)
        v2 = jnp.expand_dims(v2, axis=-1)
        v3 = jnp.expand_dims(v3, axis=-1)

        # Create an array of gradient contributions without weights
        grad_contributions = jnp.array([
            grad0[u1, u2, u3],
            grad0[u1 + 1, u2, u3],
            grad0[u1, u2 + 1, u3],
            grad0[u1, u2, u3 + 1],
            grad0[u1 + 1, u2 + 1, u3],
            grad0[u1 + 1, u2, u3 + 1],
            grad0[u1, u2 + 1, u3 + 1],
            grad0[u1 + 1, u2 + 1, u3 + 1]
        ])
        
        # Compute the maximum and minimum across all contributions
        gradxmax = jnp.max(grad_contributions, axis=0)
        gradxmin = jnp.min(grad_contributions, axis=0)
    
        # Initialize gradx
        gradx = jnp.zeros((X.shape[0], 3))

        # Compute contributions all at once
        gradx += (grad0[u1, u2, u3] * (1 - v1) * (1 - v2) * (1 - v3) * (1 + v1/2 + v2/2 + v3/2 - v1*v1/2 - v2*v2/2 - v3*v3/2))
        gradx += (grad0[u1 + 1, u2, u3] * v1 * (1 - v2) * (1 - v3) * (1 + v1/2 + v2/2 + v3/2 - v1*v1/2 - v2*v2/2 - v3*v3/2))
        gradx += (grad0[u1, u2 + 1, u3] * (1 - v1) * v2 * (1 - v3) * (1 + v1/2 + v2/2 + v3/2 - v1*v1/2 - v2*v2/2 - v3*v3/2))
        gradx += (grad0[u1, u2, u3 + 1] * (1 - v1) * (1 - v2) * v3 * (1 + v1/2 + v2/2 + v3/2 - v1*v1/2 - v2*v2/2 - v3*v3/2))
        gradx += (grad0[u1 + 1, u2 + 1, u3] * v1 * v2 * (1 - v3) * (1 + v1/2 + v2/2 + v3/2 - v1*v1/2 - v2*v2/2 - v3*v3/2))
        gradx += (grad0[u1 + 1, u2, u3 + 1] * v1 * (1 - v2) * v3 * (1 + v1/2 + v2/2 + v3/2 - v1*v1/2 - v2*v2/2 - v3*v3/2))
        gradx += (grad0[u1, u2 + 1, u3 + 1] * (1 - v1) * v2 * v3 * (1 + v1/2 + v2/2 + v3/2 - v1*v1/2 - v2*v2/2 - v3*v3/2))
        gradx += (grad0[u1 + 1, u2 + 1, u3 + 1] * v1 * v2 * v3 * (1 + v1/2 + v2/2 + v3/2 - v1*v1/2 - v2*v2/2 - v3*v3/2))

        gradx += (grad0[u1 - 1, u2, u3] * (1 - v1) * (1 - v2) * (1 - v3) * (-v1 * (2 - v1) / 6))
        gradx += (grad0[u1 + 2, u2, u3] * (1 - v1) * (1 - v2) * (1 - v3) * (-v1 * (1 + v1) / 6))
        gradx += (grad0[u1 - 1, u2 + 1, u3] * (1 - v1) * v2 * (1 - v3) * (-v1 * (2 - v1) / 6))
        gradx += (grad0[u1 + 2, u2 + 1, u3] * (1 - v1) * v2 * (1 - v3) * (-v1 * (1 + v1) / 6))
        gradx += (grad0[u1 - 1, u2, u3 + 1] * (1 - v1) * (1 - v2) * v3 * (-v1 * (2 - v1) / 6))
        gradx += (grad0[u1 + 2, u2, u3 + 1] * (1 - v1) * (1 - v2) * v3 * (-v1 * (1 + v1) / 6))
        gradx += (grad0[u1 - 1, u2 + 1, u3 + 1] * (1 - v1) * v2 * v3 * (-v1 * (2 - v1) / 6))
        gradx += (grad0[u1 + 2, u2 + 1, u3 + 1] * (1 - v1) * v2 * v3 * (-v1 * (1 + v1) / 6))
        
        gradx += (grad0[u1, u2 - 1, u3] * (1 - v1) * (1 - v2) * (1 - v3) * (-v2 * (2 - v2) / 6))
        gradx += (grad0[u1, u2 + 2, u3] * (1 - v1) * (1 - v2) * (1 - v3) * (-v2 * (1 + v2) / 6))
        gradx += (grad0[u1 + 1, u2 - 1, u3] * v1 * (1 - v2) * (1 - v3) * (-v2 * (2 - v2) / 6))
        gradx += (grad0[u1 + 1, u2 + 2, u3] * v1 * (1 - v2) * (1 - v3) * (-v2 * (1 + v2) / 6))
        gradx += (grad0[u1, u2 - 1, u3 + 1] * (1 - v1) * (1 - v2) * v3 * (-v2 * (2 - v2) / 6))
        gradx += (grad0[u1, u2 + 2, u3 + 1] * (1 - v1) * (1 - v2) * v3 * (-v2 * (1 + v2) / 6))
        gradx += (grad0[u1 + 1, u2 - 1, u3 + 1] * v1 * (1 - v2) * v3 * (-v2 * (2 - v2) / 6))
        gradx += (grad0[u1 + 1, u2 + 2, u3 + 1] * v1 * (1 - v2) * v3 * (-v2 * (1 + v2) / 6))
        
        gradx += (grad0[u1, u2, u3 - 1] * (1 - v1) * (1 - v2) * (1 - v3) * (-v3 * (2 - v3) / 6))
        gradx += (grad0[u1, u2, u3 + 2] * (1 - v1) * (1 - v2) * (1 - v3) * (-v3 * (1 + v3) / 6))
        gradx += (grad0[u1 + 1, u2, u3 - 1] * v1 * (1 - v2) * (1 - v3) * (-v3 * (2 - v3) / 6))
        gradx += (grad0[u1 + 1, u2, u3 + 2] * v1 * (1 - v2) * (1 - v3) * (-v3 * (1 + v3) / 6))
        gradx += (grad0[u1, u2 + 1, u3 - 1] * (1 - v1) * v2 * (1 - v3) * (-v3 * (2 - v3) / 6))
        gradx += (grad0[u1, u2 + 1, u3 + 2] * (1 - v1) * v2 * (1 - v3) * (-v3 * (1 + v3) / 6))
        gradx += (grad0[u1 + 1, u2 + 1, u3 - 1] * v1 * v2 * (1 - v3) * (-v3 * (2 - v3) / 6))
        gradx += (grad0[u1 + 1, u2 + 1, u3 + 2] * v1 * v2 * (1 - v3) * (-v3 * (1 + v3) / 6))

        gradx = jnp.clip(gradx, gradxmin, gradxmax)

    
        # Update points
        X = X + self.chi * t0 * gradx
    
        # Generate random noise
        diff = random.normal(key, (self.P, 3))
        return X + diff * jnp.sqrt(2 * t0 * self.mu)