import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import solve_ivp
import time

N = int(input("How many bodies do you want to simulate? "))  # Number of bodies
#m1, m2, m3, m4 = 1.0, 1.0, 1.0, 1.0  # Masses of the bodies
masses_array = np.full(N, 1)  # Masses of the bodies in an array

#print(masses_array)

def random_3D_vector():
    """Generate a random 3D vector."""
    return np.random.rand(3)

# # Position
# initial_position_1 =  [1.0,  0.0,  1.0]
# initial_position_2 =  [1.0,  1.0,  0.0]
# initial_position_3 =  [0.0,   1.0, 1.0]
# initial_position_4 =  [0.0,   0.0, 0.0]

# # Velocity
# initial_velocity_1 =  [0.0, 0.0, -1.0]
# initial_velocity_2 =  [0.0, 0.0, 1.0]
# initial_velocity_3 =  [0.0, 0.0, -0.6]
# initial_velocity_4 =  [0.0, 0.0, 0.6]

initial_conditions = np.array([random_3D_vector() for _ in range(N*2)]).ravel()
#print(initial_conditions)

# initial_conditions = np.array([
#     random_3D_vector(), random_3D_vector(), random_3D_vector(), random_3D_vector(),
#     random_3D_vector(), random_3D_vector(), random_3D_vector(), random_3D_vector()
# ]).ravel()
#print(initial_conditions)

def system_odes(t, S_flat, *masses): #*masses allows for variable number of arguments, here we expect a touple

    S = S_flat.reshape(2*N, 3)  # Reshape S to a 8x3 array

    #p1,p2,p3,p4 = S[0], S[1], S[2], S[3] # three positions
    # dp1_dt, dp2_dt, dp3_dt, d4_dt = S[4], S[5], S[6] , S[7]# three change in positions over time = VELOCITIES
    # f1, f2, f3,f4 = dp1_dt, dp2_dt, dp3_dt, d4_dt

    

    positions = S[:N]  # Extract positions
    functions = S[N:]  # Extract velocities
    
    df_dts = np.array([])
    
    for index_i, pos in enumerate(positions): #
        dfi_dt = 0
        for index_j, mas in enumerate(masses):
            if index_i != index_j:
                
                dfi_dt += mas * (positions[index_j] - pos) / np.linalg.norm(positions[index_j] - pos)**3
        df_dts = np.append(df_dts, dfi_dt)

    return np.append( functions, df_dts)
        
#system_odes(0, initial_conditions, m1, m2, m3, m4)  # Test the function to ensure it works

time_s, time_e = 0, 10 #between 0 and 10 dimentionless time seconds
t_points = np.linspace(time_s, time_e, 1001) # want to evaluate over 1001 points

solution = solve_ivp(
    fun=system_odes, # ravel() flattens the array
    t_span=(time_s, time_e),
    y0=initial_conditions, 
    t_eval=t_points,
    args=tuple(masses_array)
)


# Define some default colors (repeat if N > len(colors))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Use modern colormap handling (Matplotlib >= 3.7)
cmap = plt.colormaps['tab10'].resampled(N)

for i in range(N):
    x = solution.y[i * 3]
    y = solution.y[i * 3 + 1]
    z = solution.y[i * 3 + 2]

    color = cmap(i / N)  # float between 0 and 1
    ax.plot(x, y, z, color=color, label=f'Planet {i+1}', linewidth=1)
    ax.plot([x[-1]], [y[-1]], [z[-1]], 'o', color=color, markersize=7)

ax.set_title(f"The {N}-Body Problem")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.grid()
plt.legend()
plt.show()