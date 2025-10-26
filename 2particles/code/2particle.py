import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Parámetros físicos ---
hbar = 1.0   # constante reducida de Planck
m = 1.0      # masa de la partícula
L = 1.0      # longitud de la caja
omega = np.pi / L

# --- Dominio espacial ---
N = 100
x = np.linspace(0, L, N)
X1, X2 = np.meshgrid(x, x)

# --- Función de onda espacial ---
def psi_spatial(x1, x2, n1, n2, L=1.0):
    """Parte espacial antisimétrica de la función de onda de dos fermiones."""
    term1 = np.sin(n1 * np.pi * x1 / L) * np.sin(n2 * np.pi * x2 / L)
    term2 = np.sin(n2 * np.pi * x1 / L) * np.sin(n1 * np.pi * x2 / L)
    return (term1 - term2) / np.sqrt(2)

def prob_density(x1, x2, n1, n2, L=1.0):
    """Densidad de probabilidad |Ψ|²."""
    psi = psi_spatial(x1, x2, n1, n2, L)
    return np.abs(psi)**2

# --- Configuraciones (n1, n2) ---
configs = [(1,2), (1,3), (2,3), (2,4), (3,4), (1,4)]

# --- Crear figura con subplots 3D ---
fig = plt.figure(figsize=(12, 8))

for i, (n1, n2) in enumerate(configs, 1):
    ax = fig.add_subplot(2, 3, i, projection='3d')
    Z = prob_density(X1, X2, n1, n2, L)
    ax.plot_surface(X1, X2, Z, cmap='inferno', edgecolor='none', rstride=2, cstride=2)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$|\Psi|^2$')
    ax.set_title(f"(n₁, n₂) = ({n1}, {n2})")
    ax.view_init(elev=30, azim=45)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# --- Guardar en PDF ---
plt.savefig("fermions_3D_collage.pdf", bbox_inches='tight')
plt.close()

print("PDF file generated: fermions_3D_collage.pdf")
