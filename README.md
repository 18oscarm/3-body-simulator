This script simulates the trajectories of N massive bodies travelling through space. The complex forces exhibited by each body in space cause chaotic trajectories.

Run Nbody.py and select the number of planetary objects you want to model. 
Currently, the model plots planetary objects all with mass 1, from random starting positions and from random initial velocities. All of this can be easily changed when needed.

This was inspired by https://github.com/Younes-Toumi/Younes-Toumi

- Created a Python script to model the complex paths of multiple objects in space.

- Improved upon an existing 3-body problem script by automating and generalizing the code to handle a user-defined number of bodies.

- Applied numerical methods and scientific computing libraries, including SciPy for solving a system of ordinary differential equations (ODEs) and NumPy for managing physical data.

- Utilised Matplotlib to create a 3D visualization of the simulated planetary trajectories.

Future improvements:
- Improve efficiency
- Use other methods that don't use system_odes
- Animate the trajectories 
