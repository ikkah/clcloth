OpenCL Cloth Simulation

All the C++ code is in a single file (main.cpp), and the OpenCL code file
(kernel.cl) is loaded dynamically at runtime. To build, simply build main.cpp
and link the OpenCL (version >= 1.1), OpenGL and GLUT libraries.

For example, on OS X:

$ g++ -framework OpenCL -framework OpenGL -framework GLUT main.cpp -o main

Keys to try:
- Enter: pause/resume the simulation
- Space: run a single physics step (while paused)
- R: Reset simulation
- N: Show surface normals for the cloth
- WASD, QZ: Move camera
- C: Reset camera

For more control, tweak the parameters in config.h.
