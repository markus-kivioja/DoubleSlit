import numpy as np
import cupy as cp
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation

N = 256
h = 1 / N
dt = h * 0.001

take_timestep = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void take_timestep(complex<float>* next_psi, const complex<float>* prev_psi, const float* V, float h, float dt, int N) {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int idx = y * N + x;

        if ((x == 0) || (x == (N - 1)) || (y == 0) || (y == (N - 1)) || (V[idx] > 99999))
        {
            next_psi[idx] = 0;
            return;
        } 

        // Indices to neighboring elements
        int up_idx = (y + 1) * N + x;
        int down_idx = (y - 1) * N + x;
        int left_idx = y * N + (x - 1);
        int right_idx = y * N + (x + 1);
        
        // Compute the Laplacian using finite difference method
        complex<float> Laplacian = (prev_psi[right_idx] + prev_psi[left_idx] + 
                     prev_psi[up_idx] + prev_psi[down_idx] - 
                     4.0f * prev_psi[idx]) / (h * h);

        // The Hamiltonian
        complex<float> Hpsi = -0.5f * Laplacian + V[idx] * prev_psi[idx];

        // Time integrate using central difference method
        next_psi[idx] += dt * complex<float>(0, -1) * Hpsi;
    }
''', 'take_timestep')

comp_normSq_phase = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void normSq_phase(float* normSq, float* phase, const complex<float>* psi, int N) {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;

        if ((x == 0) || (x == (N - 1)) || (y == 0) || (y == (N - 1)))
        {
            return;
        } 
        
        int idx = y * N + x;

        normSq[idx] = (conj(psi[idx]) * psi[idx]).real();
        phase[idx] = arg(psi[idx]);
    }
''', 'normSq_phase')

def integrate_normSq(psi):
    result = 0
    for x in range(N):
        for y in range(N):
            result += h * h * (np.conj(psi[y, x]) * psi[y, x]).real
    return result

# Define GPU thread block and block group sizes
block_size = (8, 8,)
grid_size = (math.ceil(N / block_size[0]), math.ceil(N / block_size[1]),)

# Initialize the wave function and potential
even_psi_h = np.zeros((N, N), dtype=np.complex64)
odd_psi_h = np.zeros((N, N), dtype=np.complex64)
V_h = np.zeros((N, N), dtype=np.float32)
sigma = 1 / 12.
pos0_x = 3 * sigma
pos0_y = 0.5
k = -18.0 * 2*np.pi
for x_id in range(N):
    for y_id in range(N):
        x = x_id * h
        y = y_id * h
        diff_x = x - pos0_x
        diff_y = y - pos0_y
        psi = np.exp(-(diff_x*diff_x + diff_y*diff_y) / (2 * sigma * sigma))
        phase = np.sin(k*x) + 1j*np.cos(k*x)
        even_psi_h[y_id, x_id] = psi * phase
        odd_psi_h[y_id, x_id] = psi * phase

normSq = integrate_normSq(even_psi_h)
even_psi_h /= np.sqrt(normSq)
odd_psi_h /= np.sqrt(normSq)
normSq = integrate_normSq(odd_psi_h)
print(normSq)

V_h[0:int(N * 0.4), int(N * 0.8):int(N * 0.83)] = 999999.0
V_h[int(N * 0.45):int(N * 0.55), int(N * 0.8):int(N * 0.83)] = 999999.0
V_h[int(N * 0.6):N, int(N * 0.8):int(N * 0.83)] = 999999.0

# Copy data from CPU to GPU
even_psi_d = cp.array(even_psi_h.reshape(-1), dtype=cp.complex64)
odd_psi_d = cp.array(odd_psi_h.reshape(-1), dtype=cp.complex64)
V_d = cp.array(V_h.reshape(-1), dtype=cp.float32)

# Initialize the visualization
normSq_d = cp.zeros(N*N, dtype=cp.float32)
phase_d = cp.zeros(N*N, dtype=cp.float32)
fig, ax = plt.subplots()
comp_normSq_phase(grid_size, block_size, (normSq_d, phase_d, even_psi_d, cp.int32(N)))
im = ax.imshow(normSq_d.get().reshape((N, N)), animated=True)

def time_integrate(i, *args):
    # Update an odd time step
    take_timestep(grid_size, block_size, (odd_psi_d, even_psi_d, V_d, cp.float32(h), cp.float32(dt), cp.int32(N)))
    # Update an even time step
    take_timestep(grid_size, block_size, (even_psi_d, odd_psi_d, V_d, cp.float32(h), cp.float32(dt), cp.int32(N)))

    # Compute the probability/phase and visualize
    comp_normSq_phase(grid_size, block_size, (normSq_d, phase_d, even_psi_d, cp.int32(N)))
    im.set_data(V_h + normSq_d.get().reshape((N, N)))
    return im,

ani = animation.FuncAnimation(fig, time_integrate, interval=1, blit=True)
plt.show()