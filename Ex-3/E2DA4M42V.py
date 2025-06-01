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
    def __init__(self,M0=25,H=256,L=8,reg=0,chi=1,mu=1,rng_key=random.PRNGKey(216)):
        start_time = time.time()
        self.H=2*int(H/2)
        self.L=L
        self.reg=reg
        self.chi=chi
        self.mu=mu
        self.key=rng_key
        self.M0=M0
        self.mlen = self.L / self.H
        self.Cv=jnp.zeros((self.H,self.H))
        self.varlog=[]
        self.tlog=[]
        self.cmaxlog=[]
        self.key, subkey = random.split(self.key)
        print("--- Build time %s s ---" % (time.time() - start_time))
        return

    def solve(self,t,t0,show_num=5,show_flag=True,notebook_name="simulation"):

        self.P = self.X.shape[0]
        self.M = self.M0 / self.P
        self.offset = self.H / 2
        start_time0 = time.time()
        nt = int(t/t0)
        u1 = jnp.arange(self.H).reshape(self.H, 1)
        u2 = jnp.arange(self.H).reshape(1, self.H)
        y2 = (4*(jnp.pi**2)/(self.L**2)) * ((u1-self.offset)**2+(u2-self.offset)**2)
        self.FK = jnp.where(y2==0, 0, -1/y2)
        coords = jnp.indices((self.H, self.H)).transpose(1, 2, 0)
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

            if (it + 1) % show_num == 0 or it == nt - 1:
                if show_flag:
                    current_time = (it + 1) * t0
                    formatted_time = f"{current_time:.6f}".replace('.', '_')
                    filename = f"2Ddata/{notebook_name}_{formatted_time}.npy"
                    x_np = np.array(self.X)
                    np.save(filename, x_np)


        
        print("--- Total time %s s ---" % (time.time() - start_time0))
        return

    
    @partial(jit, static_argnums=(0,))
    def Cstep(self, X, Cv, t0):
        # Compute positions
        u_pos = jnp.clip((X + self.L / 2) / self.mlen, 1 + 1e-8, self.H - 1 - 1e-8)
        u_int = jnp.floor(u_pos).astype(jnp.int32)
        u_frac = u_pos - u_int
    
        # Prepare for contributions
        u1, u2 = u_int[:, 0], u_int[:, 1]
        v1, v2 = u_frac[:, 0], u_frac[:, 1]
    
        # Initialize rho
        rho = jnp.zeros((self.H, self.H))
        
        # Compute contributions using broadcasting
        rho = rho.at[u1, u2].add((1 - v1) * (1 - v2) * (1 + v1/2 + v2/2 - v1*v1/2 - v2*v2/2))
        rho = rho.at[u1 + 1, u2].add(v1 * (1 - v2) * (1 + v1/2 + v2/2 - v1*v1/2 - v2*v2/2))
        rho = rho.at[u1, u2 + 1].add((1 - v1) * v2 * (1 + v1/2 + v2/2 - v1*v1/2 - v2*v2/2))
        rho = rho.at[u1 + 1, u2 + 1].add(v1 * v2 * (1 + v1/2 + v2/2 - v1*v1/2 - v2*v2/2))

        rho = rho.at[u1 - 1, u2].add(-(1 - v1) * (1 - v2) * v1 * (2 - v1) / 6)
        rho = rho.at[u1 + 2, u2].add(-(1 - v1) * (1 - v2) * v1 * (1 + v1) / 6)
        rho = rho.at[u1 - 1, u2 + 1].add(-(1 - v1) * v2 * v1 * (2 - v1) / 6)
        rho = rho.at[u1 + 2, u2 + 1].add(-(1 - v1) * v2 * v1 * (1 + v1) / 6)
        rho = rho.at[u1, u2 - 1].add(-(1 - v1) * (1 - v2) * v2 * (2 - v2) / 6)
        rho = rho.at[u1, u2 + 2].add(-(1 - v1) * (1 - v2) * v2 * (1 + v2) / 6)
        rho = rho.at[u1 + 1, u2 - 1].add(-v1 * (1 - v2) * v2 * (2 - v2) / 6)
        rho = rho.at[u1 + 1, u2 + 2].add(-v1 * (1 - v2) * v2 * (1 + v2) / 6) 

    
        # Perform FFT
        rhov = jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.fftshift(self.M * rho)))
        
        # Compute the convective term
        conv_term = -rhov
        
        return self.FK * conv_term

    @partial(jit, static_argnums=(0,))
    def Xstep(self, X, Cv, t0, key):
        Fgrad = self.FN * jnp.expand_dims(Cv, axis=-1)
        
        # Compute gradients
        grad0 = jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.fftshift(Fgrad, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))) / (self.mlen**2)

    
        # Compute positions and contributions
        u_pos = jnp.clip((X + self.L / 2) / self.mlen, 1 + 1e-8, self.H - 1 - 1e-8)
        u_int = jnp.floor(u_pos).astype(jnp.int32)
        u_frac = u_pos - u_int
    
        u1, u2 = u_int[:, 0], u_int[:, 1]
        v1, v2 = u_frac[:, 0], u_frac[:, 1]

        v1 = jnp.expand_dims(v1, axis=-1)
        v2 = jnp.expand_dims(v2, axis=-1)
    
        # Initialize gradx
        gradx = jnp.zeros((X.shape[0], 2))

        # Compute contributions all at once
        gradx += (grad0[u1, u2] * (1 - v1) * (1 - v2))
        gradx += (grad0[u1 + 1, u2] * v1 * (1 - v2))
        gradx += (grad0[u1, u2 + 1] * (1 - v1) * v2)
        gradx += (grad0[u1 + 1, u2 + 1] * v1 * v2)


    
        # Update points
        X = X + self.chi * t0 * gradx
    
        # Generate random noise
        diff = random.normal(key, (self.P, 2))
        return X + diff * jnp.sqrt(2 * t0 * self.mu)