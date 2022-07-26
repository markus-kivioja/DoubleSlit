import numpy as np
import cupy as cp
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import widgets

N = 256 # Number of grid nodes per dimension
h = 1 / N # Length between neighbour grid nodes
dt = 0.14 * h*h # The greatest numerically stable time step size
V_wall = 200000.0 # The greatest numerically stable potential for walls

inv_hSq = 1.0 / (h*h) # For performance optimization

# GPU code for one central-difference time step for every grid node simultaneously
take_timestep = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void take_timestep(complex<float>* next_psi, const complex<float>* prev_psi, const float* V, float inv_hSq, float dt, int N) {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int idx = y * N + x;

        // Dirichlet boundary condition: \psi = 0
        if ((x == 0) || (x == (N - 1)) || (y == 0) || (y == (N - 1)))
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
                     4.0f * prev_psi[idx]) * inv_hSq;

        // The Hamiltonian
        complex<float> Hpsi = -0.5f * Laplacian + V[idx] * prev_psi[idx];

        // Time integrate using central difference method
        next_psi[idx] += 2.0f * dt * complex<float>(0, -1) * Hpsi;
    }
''', 'take_timestep')

# GPU code for one forward Euler time step for every grid node simultaneously
forward_euler = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void forward_euler(complex<float>* next_psi, const complex<float>* prev_psi, const float* V, float inv_hSq, float dt, int N) {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int idx = y * N + x;

        // Dirichlet boundary condition: \psi = 0
        if ((x == 0) || (x == (N - 1)) || (y == 0) || (y == (N - 1)))
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
                     4.0f * prev_psi[idx]) * inv_hSq;

        // The Hamiltonian
        complex<float> Hpsi = -0.5f * Laplacian + V[idx] * prev_psi[idx];

        // Time integrate using central difference method
        next_psi[idx] = prev_psi[idx] + dt * complex<float>(0, -1) * Hpsi;
    }
''', 'forward_euler')

# GPU code for computing the norm squared values of psi elements
comp_normSq_phase = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void normSq_phase(float* normSq, float* phase, const complex<float>* psi, int N) {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
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

odd_psi_h = np.zeros((N, N), dtype=np.complex64)
V_h = np.zeros((N, N), dtype=np.float32)

# Initialize the wave function with a wave packet
radius = 1 / 11.
pos0_x = 3 * radius
pos0_y = 0.5
momentum = -18.0 * 2*np.pi
for x_id in range(N):
    for y_id in range(N):
        x = x_id * h
        y = y_id * h
        diff_x = x - pos0_x
        diff_y = y - pos0_y
        psi = np.exp(-(diff_x*diff_x + diff_y*diff_y) / (2 * radius * radius))
        phase = np.sin(momentum*x) + 1j*np.cos(momentum*x)
        odd_psi_h[y_id, x_id] = psi * phase

# Normalize the wave function
normSq = integrate_normSq(odd_psi_h)
odd_psi_h /= np.sqrt(normSq)

# Copy data from CPU to GPU
odd_psi_d = cp.array(odd_psi_h.reshape(-1), dtype=cp.complex64)

# Add the 2x5 lattice potential
def add_v_lattice():
    global V_h
    global V_d
    wall_width = 0.08
    wall_height = 0.01
    wall_gap_y = 0.05
    wall_gap_x = 0.071
    wall_offset_y = 0.5 - (5 * wall_width + 4 * wall_gap_y) * 0.5
    wall_offset_x = 0.75
    for x in range(2):
        for y in range(5):
            y_start = wall_offset_y + y * (wall_width + wall_gap_y)
            y_end = y_start + wall_width
            x_start = wall_offset_x + x * (wall_height + wall_gap_x)
            x_end = x_start + wall_height
            V_h[math.ceil(y_start*N):math.ceil(y_end*N), math.ceil(x_start*N):math.ceil(x_end*N)] = V_wall

    V_d = cp.array(V_h.reshape(-1), dtype=cp.float32)

add_v_lattice()

# Take one forward Euler step to form the other needed initial value
even_psi_d = cp.zeros(N*N, dtype=cp.complex64)
forward_euler(grid_size, block_size, (even_psi_d, odd_psi_d, V_d, cp.float32(inv_hSq), cp.float32(dt), cp.int32(N)))

iters_per_frame = 2
is_playing = False

### The main loop - central difference time integration of the Schrödinger equation ###
def time_integrate(i, *args):
    if is_playing:
        for _ in range(iters_per_frame):
            # Update an odd time step
            take_timestep(grid_size, block_size, (odd_psi_d, even_psi_d, V_d, cp.float32(inv_hSq), cp.float32(dt), cp.int32(N)))
            # Update an even time step
            take_timestep(grid_size, block_size, (even_psi_d, odd_psi_d, V_d, cp.float32(inv_hSq), cp.float32(dt), cp.int32(N)))

    # Compute the probability/phase and visualize
    comp_normSq_phase(grid_size, block_size, (normSq_d, phase_d, even_psi_d, cp.int32(N)))
    im.set_data(V_d.get().reshape((N, N)) + normSq_d.get().reshape((N, N)))
    return im,




############ The user interface and visualization ############
normSq_d = cp.zeros(N*N, dtype=cp.float32)
phase_d = cp.zeros(N*N, dtype=cp.float32)
fig, ax = plt.subplots()
comp_normSq_phase(grid_size, block_size, (normSq_d, phase_d, even_psi_d, cp.int32(N)))
im = ax.imshow(normSq_d.get().reshape((N, N)), animated=True)
left_is_down = False
right_is_down = False

def play_toggle_clicked(event):
    global is_playing
    is_playing = not is_playing
    if is_playing:
        play_toggle_button.label.set_text("Pause")
    else:
        play_toggle_button.label.set_text("Play")
    play_toggle_button.canvas.draw_idle()
def reset_state_clicked(event):
    global even_psi_d, odd_psi_d
    odd_psi_d = cp.array(odd_psi_h.reshape(-1), dtype=cp.complex64)
    forward_euler(grid_size, block_size, (even_psi_d, odd_psi_d, V_d, cp.float32(inv_hSq), cp.float32(dt), cp.int32(N)))
def clear_v_clicked(event):
    global V_h
    global V_d
    V_h[0:N, 0:N] = 0.0
    V_d = cp.array(V_h.reshape(-1), dtype=cp.float32)
def add_lattice_clicked(event):
    add_v_lattice()
def mouse_down(event):
    global left_is_down
    global right_is_down
    global V_d
    if event.xdata is None or event.ydata is None or event.xdata < 1 or event.ydata < 1:
        return
    if event.button == mpl.backend_bases.MouseButton.LEFT:
        left_is_down = True
    elif event.button == mpl.backend_bases.MouseButton.RIGHT:
        right_is_down = True
    x = int(event.xdata)
    y = int(event.ydata)
    if left_is_down:
        for x_offset in range(-1,2):
            for y_offset in range(0,3):
                V_h[y + y_offset, x + x_offset] = V_wall
        V_d = cp.array(V_h.reshape(-1), dtype=cp.float32)
    elif right_is_down:
        for x_offset in range(-2,3):
            for y_offset in range(-2,3):
                V_h[y + y_offset, x + x_offset] = 0.0
        V_d = cp.array(V_h.reshape(-1), dtype=cp.float32)
def mouse_up(event):
    global left_is_down
    global right_is_down
    if event.button == mpl.backend_bases.MouseButton.LEFT:
        left_is_down = False
    elif event.button == mpl.backend_bases.MouseButton.RIGHT:
        right_is_down = False
def mouse_moves(event):
    global left_is_down
    global right_is_down
    global V_d
    if event.xdata is None or event.ydata is None or event.xdata < 1 or event.ydata < 1:
        return
    x = int(event.xdata)
    y = int(event.ydata)
    if left_is_down:
        for x_offset in range(-1,2):
            for y_offset in range(0,3):
                V_h[y + y_offset, x + x_offset] = V_wall
        V_d = cp.array(V_h.reshape(-1), dtype=cp.float32)
    elif right_is_down:
        for x_offset in range(-2,3):
            for y_offset in range(-2,3):
                V_h[y + y_offset, x + x_offset] = 0.0
        V_d = cp.array(V_h.reshape(-1), dtype=cp.float32)

button_width = 0.17
button_gap = 0.01
button_left = 0.15
play_toggle_button_ax = plt.axes([button_left, 0.9, button_width, 0.075])
play_toggle_button = widgets.Button(play_toggle_button_ax, "Play")
play_toggle_button.on_clicked(play_toggle_clicked)
button_left += button_width + button_gap
reset_state_button_ax = plt.axes([button_left, 0.9, button_width, 0.075])
reset_state_button = widgets.Button(reset_state_button_ax, "Reset particle")
reset_state_button.on_clicked(reset_state_clicked)
button_left += button_width + button_gap
clear_v_button_ax = plt.axes([button_left, 0.9, button_width, 0.075])
clear_v_button = widgets.Button(clear_v_button_ax, "Clear potential")
clear_v_button.on_clicked(clear_v_clicked)
button_left += button_width + button_gap
add_double_slit_ax = plt.axes([button_left, 0.9, button_width, 0.075])
add_double_slit_button = widgets.Button(add_double_slit_ax, "Add lattice")
add_double_slit_button.on_clicked(add_lattice_clicked)
fig.canvas.mpl_connect('button_press_event', mouse_down)
fig.canvas.mpl_connect('button_release_event', mouse_up)
fig.canvas.mpl_connect('motion_notify_event', mouse_moves)

ani = animation.FuncAnimation(fig, time_integrate, interval=0, blit=True)

# Start the program
plt.show()