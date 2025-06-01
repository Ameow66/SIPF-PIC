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
import os

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

    def solve(self,t,t0,show_num=100,show_flag=True):

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

        os.makedirs('3d_plots', exist_ok=True)
        os.makedirs('scatterplots', exist_ok=True)
        os.makedirs('modulus_distributions', exist_ok=True)
        
        for it in range(nt):
            self.key, subkey = random.split(self.key)
            self.X1 = self.Xstep(self.X, self.Cv, t0, subkey)



            self.Cv = self.Cstep(self.X, self.Cv, t0)

            
            self.X = self.X1
            self.varlog.append(jnp.var(self.X))
            self.tlog.append((it+1)*t0)
            self.cmaxlog.append(jnp.max(jnp.abs(jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.fftshift(self.Cv))))))

             # Visualization logic
            if show_flag and (it + 1) % show_num == 0:
                X_np = jax.device_get(self.X)
                
                
                fig1 = plt.figure(figsize=(6, 6))
                ax1 = fig1.add_subplot(111, projection='3d')
                ax1.scatter(X_np[:20000, 0], X_np[:20000, 1], X_np[:20000, 2], 
                            s=1, alpha=0.6, c='b', marker='o')
              
                plt.tight_layout()
                plt.savefig(f'3d_plots/3d_plot_iter_{it+1}.png')
                plt.close(fig1)
                
                              
                fig2 = plt.figure(figsize=(4, 4))
                ax2 = fig2.add_subplot(111)

                ax2.scatter(X_np[:20000, 0],
                            X_np[:20000, 1],
                            s=1,
                            alpha=0.5,
                            c='#1f77b4',       
                            edgecolors='none',
                            linewidths=0       
                            )

                ax2.set_xlabel('$X_1$', fontsize=12)
                ax2.set_ylabel('$X_2$', fontsize=12)
                ax2.set_xlim(-1.6, 1.6)
                ax2.set_ylim(-1.6, 1.6)

          
                ax2.grid(True, 
                        color='gray', 
                        linestyle=':', 
                        linewidth=0.5,
                        alpha=0.4)

                plt.tight_layout()
                plt.savefig(f'scatterplots/scatter_iter_{it+1}.eps',format='eps', dpi=300)  
                plt.close(fig2)

                fig3 = plt.figure(figsize=(10, 7))
                ax3 = fig3.add_subplot(111)
                
               
                modulus = np.linalg.norm(X_np, axis=1)  
                bins = np.linspace(0, 1.5, 30)  
                hist, _ = np.histogram(modulus, bins=bins)
                
                
                bin_width = bins[1] - bins[0]
                density_percent = (hist / len(modulus)) / bin_width * 100
                
                
                ax3.bar(bins[:-1], density_percent, 
                       width=bin_width, 
                       align='edge', 
                       edgecolor='k',
                       alpha=0.7)
                
            
                ax3.set_xlim(0, 1.8)
                ax3.set_ylim(0, 500)  
                ax3.set_xlabel("Modulus Value")
                ax3.set_ylabel("Percentage/Density (%)")
                ax3.set_title("Modulus Distribution Density")
                ax3.grid(True, linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                plt.savefig(f'modulus_distributions/modulus_dist_iter_{it+1}.png')
                plt.close(fig3)

        
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
        u_pos = jnp.clip((X + self.L / 2) / self.mlen, 1 + 1e-8, self.H - 1 - 1e-8)
        u_int = jnp.floor(u_pos).astype(jnp.int32)
        u_frac = u_pos - u_int
    
        u1, u2, u3 = u_int[:, 0], u_int[:, 1], u_int[:, 2]
        v1, v2, v3 = u_frac[:, 0], u_frac[:, 1], u_frac[:, 2]

        v1 = jnp.expand_dims(v1, axis=-1)
        v2 = jnp.expand_dims(v2, axis=-1)
        v3 = jnp.expand_dims(v3, axis=-1)

    
        # Initialize gradx
        gradx = jnp.zeros((X.shape[0], 3))

        # Compute contributions all at once
        gradx += (grad0[u1, u2, u3] * (1 - v1) * (1 - v2) * (1 - v3))
        gradx += (grad0[u1 + 1, u2, u3] * v1 * (1 - v2) * (1 - v3))
        gradx += (grad0[u1, u2 + 1, u3] * (1 - v1) * v2 * (1 - v3))
        gradx += (grad0[u1, u2, u3 + 1] * (1 - v1) * (1 - v2) * v3)
        gradx += (grad0[u1 + 1, u2 + 1, u3] * v1 * v2 * (1 - v3))
        gradx += (grad0[u1 + 1, u2, u3 + 1] * v1 * (1 - v2) * v3)
        gradx += (grad0[u1, u2 + 1, u3 + 1] * (1 - v1) * v2 * v3)
        gradx += (grad0[u1 + 1, u2 + 1, u3 + 1] * v1 * v2 * v3)

    
        # Update points
        X = X + self.chi * t0 * gradx
    
        # Generate random noise
        diff = random.normal(key, (self.P, 3))
        return X + diff * jnp.sqrt(2 * t0 * self.mu)