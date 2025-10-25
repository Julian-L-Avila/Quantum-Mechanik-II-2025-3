import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Physical parameters ---
hbar = 1.0   # reduced Planck constant
m = 1.0      # particle mass
L = 1.0      # length of the box
omega = np.pi / L

# --- Spatial domain ---
N = 100
x = np.linspace(0, L, N)
X1, X2 = np.meshgrid(x, x)

# --- Spatial wavefunction ---
def psi_spatial(x1, x2, n, k, L=1.0):
    """Antisymmetric spatial part of the two-fermion wavefunction."""
    term1 = np.sin(n * np.pi * x1 / L) * np.sin(k * np.pi * x2 / L)
    term2 = np.sin(k * np.pi * x1 / L) * np.sin(n * np.pi * x2 / L)
    return (term1 - term2) / np.sqrt(2)

def prob_density(x1, x2, n, k, L=1.0):
    """Probability density |Ψ|²."""
    psi = psi_spatial(x1, x2, n, k, L)
    return np.abs(psi)**2

# --- (n, k) configurations ---
configs = [(1,2), (1,3), (2,3), (2,4), (3,4), (1,4)]

# --- Create figure with 3D subplots ---
fig = plt.figure(figsize=(12, 8))
fig.suptitle("Probability Density |Ψ_F(x₁, x₂)|² for Different (n, k) Configurations", fontsize=14)

for i, (n, k) in enumerate(configs, 1):
    ax = fig.add_subplot(2, 3, i, projection='3d')
    Z = prob_density(X1, X2, n, k, L)
    ax.plot_surface(X1, X2, Z, cmap='inferno', edgecolor='none', rstride=2, cstride=2)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$|\Psi|^2$')
    ax.set_title(f"(n, k) = ({n}, {k})")
    ax.view_init(elev=30, azim=45)  # default camera view

plt.tight_layout(rect=[0, 0, 1, 0.96])

# --- Save to PDF ---
plt.savefig("fermions_3D_collage.pdf", bbox_inches='tight')
plt.close()

print("PDF file generated: fermions_3D_collage.pdf")
