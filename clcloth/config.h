#ifndef CONFIG_H
#define CONFIG_H


// 32x32 on sphere
#if 0
#define CLOTH_SIZE 32
#define SOLVER_ITERATIONS 1
#define SOLVER_DAMPING 0.0f
#define ENABLE_SPHERE_COLLISION 1
#endif

// 128x128 on sphere
#if 1
#define CLOTH_SIZE 128
#define SOLVER_ITERATIONS 9
//#define SOLVER_DAMPING 0.0f
#define SOLVER_DAMPING 0.02f
#define ENABLE_SPHERE_COLLISION 1
#endif

// 128x128 on round table
#if 0
#define CLOTH_SIZE 128
#define SOLVER_ITERATIONS 9
#define SOLVER_DAMPING 0.02f
#define ENABLE_CYLINDER_COLLISION 1
#endif

// 128x128 on cube
#if 0
#define CLOTH_SIZE 128
#define SOLVER_ITERATIONS 9
#define SOLVER_DAMPING 0.02f
#define ENABLE_CUBE_COLLISION 1
#endif


// device settings
#if 0
// cpu
#define BLOCK_SIZE 1
#define DEVICE_TYPE CL_DEVICE_TYPE_CPU
#else
// gpu
#define BLOCK_SIZE 8
#define DEVICE_TYPE CL_DEVICE_TYPE_GPU
#define USE_LOCAL_MEMORY
#endif

// collision shapes
#define CYLINDER_RADIUS 14.0f
#define CYLINDER_HEIGHT -2.0f
#define CYLINDER_THICKNESS 1.0f

#define SPHERE_RADIUS 12.0f

#define ENABLE_PLANE_COLLISION 1
#define PLANE_HEIGHT -12.0f

#define CUBE_SIZE 10.0f

// solver parameters
#define SOLVER_TIMESTEP (1.0f / 60.0f)
#define SOLVER_GRAVITY -27.7f
#define SOLVER_STIFFNESS 0.115f

// cloth parameters
#define CLOTH_START_X 0.7f
#define CLOTH_START_Y 0.7f
#define CLOTH_START_Z 14.1f
#define CLOTH_SCALE 32.0f

// app parameters
#define PHYSICS_TICS_PER_RENDER_FRAME 1
#define TARGET_FRAME_RATE 60
#define CAMERA_X 30.0f
#define CAMERA_Y 20.0f
#define CAMERA_FOV 67.5f


// internal stuff
#define BORDER 2
#define TEMP_SIZE (BLOCK_SIZE + 2 * BORDER)


#endif
