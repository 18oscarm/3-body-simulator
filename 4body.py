import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

N = 4  # Number of bodies
#m1, m2, m3, m4 = 1.0, 1.0, 1.0, 1.0  # Masses of the bodies
masses_array = np.full(N, 1)  # Masses of the bodies in an array

#print(masses_array)

def random_3D_vector():
    """Generate a random 3D vector."""
    return np.random.rand(3)

# Position
initial_position_1 =  [1.0,  0.0,  1.0]
initial_position_2 =  [1.0,  1.0,  0.0]
initial_position_3 =  [0.0,   1.0, 1.0]
initial_position_4 =  [0.0,   0.0, 0.0]

# Velocity
initial_velocity_1 =  [0.0, 0.0, -1.0]
initial_velocity_2 =  [0.0, 0.0, 1.0]
initial_velocity_3 =  [0.0, 0.0, -0.6]
initial_velocity_4 =  [0.0, 0.0, 0.6]

initial_conditions = np.array([
    initial_position_1, initial_position_2, initial_position_3, initial_position_4,
    initial_velocity_1, initial_velocity_2, initial_velocity_3, initial_velocity_4
]).ravel()


initial_conditions = np.array([
    random_3D_vector(), random_3D_vector(), random_3D_vector(), random_3D_vector(),
    random_3D_vector(), random_3D_vector(), random_3D_vector(), random_3D_vector()
]).ravel()
#print(initial_conditions)

def system_odes(t, S_flat, *masses):

    S = S_flat.reshape(2*N, 3)  # Reshape S to a 8x3 array
    p1,p2,p3,p4 = S[0], S[1], S[2], S[3] # three positions
    dp1_dt, dp2_dt, dp3_dt, d4_dt = S[4], S[5], S[6] , S[7]# three change in positions over time = VELOCITIES
    f1, f2, f3,f4 = dp1_dt, dp2_dt, dp3_dt, d4_dt

    functions = [f1, f2, f3, f4]

    positions = np.array([p1, p2, p3, p4])  
    
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


t_sol = solution.t
p1x_sol = solution.y[0]
p1y_sol = solution.y[1]
p1z_sol = solution.y[2]


p2x_sol = solution.y[3]
p2y_sol = solution.y[4]
p2z_sol = solution.y[5]

p3x_sol = solution.y[6]
p3y_sol = solution.y[7]
p3z_sol = solution.y[8]

p4x_sol = solution.y[9]
p4y_sol = solution.y[10]
p4z_sol = solution.y[11]

# -------  Plotting the solutions ------- #

fig, ax = plt.subplots(subplot_kw={"projection":"3d"})

planet1_plt, = ax.plot(p1x_sol, p1y_sol, p1z_sol, 'green', label='Planet 1', linewidth=1)
planet2_plt, = ax.plot(p2x_sol, p2y_sol, p2z_sol, 'red', label='Planet 2', linewidth=1)
planet3_plt, = ax.plot(p3x_sol, p3y_sol, p3z_sol, 'blue',label='Planet 3', linewidth=1)
planet4_plt, = ax.plot(p4x_sol, p4y_sol, p4z_sol, 'orange', label='Planet 4', linewidth=1)

planet1_dot, = ax.plot([p1x_sol[-1]], [p1y_sol[-1]], [p1z_sol[-1]], 'o', color='green', markersize=7)
planet2_dot, = ax.plot([p2x_sol[-1]], [p2y_sol[-1]], [p2z_sol[-1]], 'o', color='red', markersize=7)
planet3_dot, = ax.plot([p3x_sol[-1]], [p3y_sol[-1]], [p3z_sol[-1]], 'o', color='blue', markersize=7)
planet4_dot, = ax.plot([p4x_sol[-1]], [p4y_sol[-1]], [p4z_sol[-1]], 'o', color='orange', markersize=7)



ax.set_title("The 4-Body Problem")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.grid()
plt.legend()
plt.show()
