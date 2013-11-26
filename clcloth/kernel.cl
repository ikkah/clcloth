#include "config.h"

__kernel void advance(__global float4* old_positions,
                      __global float4* positions,
                      __global float4* unconstrained)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    size_t id = y * CLOTH_SIZE + x;
    
    if (x < 0 || y < 0 || x >= CLOTH_SIZE || y >= CLOTH_SIZE)
        return;
    
    float4 gravity = {0.0f, 0.0f, SOLVER_GRAVITY, 0.0f};
    float timestep = SOLVER_TIMESTEP;
    float damping = SOLVER_DAMPING;
    float4 vel = (2.0f - damping) * positions[id] - (1.0f - damping) * old_positions[id];
    float4 acc = gravity * timestep * timestep;
    unconstrained[id] = vel + acc;
}

float4 satisfy_constraint(float4 first, float4 second, float rest_distance)
{
    float stiffness = SOLVER_STIFFNESS;
    
    float4 delta = second - first;
    float delta_length = fast_length(delta);
    float difference = (delta_length - rest_distance) / delta_length;
    return delta * stiffness * difference;
}

#ifdef USE_LOCAL_MEMORY

#define fill(x_offset, y_offset)\
    temp[(local_y + y_offset + BORDER) + (local_x + x_offset + BORDER) * TEMP_SIZE] = unconstrained[(y + y_offset) * (CLOTH_SIZE) + (x + x_offset)];
#define lookup(x_offset, y_offset)\
    temp[(local_y + y_offset + BORDER) + (local_x + x_offset + BORDER) * TEMP_SIZE]

#else

#define fill(x_offset, y_offset)
#define lookup(x_offset, y_offset)\
    unconstrained[(y + y_offset) * (CLOTH_SIZE) + (x + x_offset)]

#endif

__kernel void constrain(__global float4* unconstrained,
                        __global float4* positions,
                        __local float4* temp)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    size_t id = y * (CLOTH_SIZE) + x;
    
    if (x < 0 || y < 0 || x >= CLOTH_SIZE || y >= CLOTH_SIZE)
        return;
    
    const float scale = CLOTH_SCALE / CLOTH_SIZE;
    
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    if (x + BORDER < CLOTH_SIZE && y + BORDER < CLOTH_SIZE)
        fill(+BORDER, +BORDER);
    if (local_x <= BORDER + 1 && local_y <= BORDER + 1 && x - BORDER >= 0 && y - BORDER >= 0)
        fill(-BORDER, -BORDER);
    if (local_y <= BORDER + 1 && x + BORDER < CLOTH_SIZE && y - BORDER >= 0)
        fill(+BORDER, -BORDER);
    if (local_x <= BORDER + 1 && x - BORDER >= 0 && y + BORDER < CLOTH_SIZE)
        fill(-BORDER, +BORDER);

    barrier(CLK_LOCAL_MEM_FENCE);
    
    float4 output = unconstrained[id];

    float4 dx = {0.0f, 0.0f, 0.0f, 0.0f};
    
	// straight distance constraints: prevents stretching
    
	const float straight_distance = 1.0f * scale;
	
	if (x > 0)
        dx += satisfy_constraint(output, lookup(-1,  0), straight_distance);
	if (x < (CLOTH_SIZE - 1))
		dx += satisfy_constraint(output, lookup(+1,  0), straight_distance);
	if (y < (CLOTH_SIZE - 1))
		dx += satisfy_constraint(output, lookup( 0, +1), straight_distance);
	if (y > 0)
		dx += satisfy_constraint(output, lookup( 0, -1), straight_distance);
    
	// diagonal distance constraints: prevents shearing
    
	const float diagonal_distance = sqrt(2.0f) * scale;
    
	if (x > 0 && y > 0)
		dx += satisfy_constraint(output, lookup(-1, -1), diagonal_distance);
	if (x < (CLOTH_SIZE - 1) && y > 0)
		dx += satisfy_constraint(output, lookup(+1, -1), diagonal_distance);
	if (x > 0 && y < (CLOTH_SIZE - 1))
		dx += satisfy_constraint(output, lookup(-1, +1), diagonal_distance);
	if (x < (CLOTH_SIZE - 1) && y < (CLOTH_SIZE - 1))
		dx += satisfy_constraint(output, lookup(+1, +1), diagonal_distance);
	
	// double diagonal distance constraints: prevents folding
	const float double_diagonal_distance = 2.0f * sqrt(2.0f) * scale;
    
	if (x > 1 && y > 1)
		dx += satisfy_constraint(output, lookup(-2, -2), double_diagonal_distance);
	if (x < (CLOTH_SIZE - 2) && y > 1)
		dx += satisfy_constraint(output, lookup(+2, -2), double_diagonal_distance);
	if (x > 1 && y < (CLOTH_SIZE - 2))
		dx += satisfy_constraint(output, lookup(-2, +2), double_diagonal_distance);
	if (x < (CLOTH_SIZE - 2) && y < (CLOTH_SIZE - 2))
		dx += satisfy_constraint(output, lookup(+2, +2), double_diagonal_distance);

	output += dx;

#ifdef ENABLE_SPHERE_COLLISION
    float sphere_radius = SPHERE_RADIUS;
    float4 sphere_position = {0.0f, 0.0f, 0.0f, 1.0f};
    
    float4 delta = sphere_position - output;
    float delta_length = fast_length(delta);
    if (delta_length < sphere_radius)
    {
        float difference = (delta_length - sphere_radius) / delta_length;
        output += delta * difference;
    }
#endif
    
#ifdef ENABLE_CYLINDER_COLLISION
    float radius = CYLINDER_RADIUS;
    float table_height = CYLINDER_HEIGHT;
    float table_width = CYLINDER_THICKNESS;
    
    float4 flat = output;
    flat.z = 0.0f;
    float flatLength = fast_length(flat);
    
    if (flatLength < radius && fabs(output.z - table_height) < table_width * 0.5f)
    {
        float topDistance = table_height + table_width * 0.5f - output.z;
        float bottomDistance = output.z - (table_height - table_width * 0.5f);
        float xyDistance = radius - flatLength;
        if (topDistance < xyDistance && topDistance < bottomDistance)
        {
            output.z = table_height + table_width * 0.5f;
        }
        else if (bottomDistance < xyDistance)
        {
            output.z = table_height - table_width * 0.5f;
        }
        else
        {
            output.x = radius * flat.x / flatLength;
            output.y = radius * flat.y / flatLength;
        }
    }
#endif
    
#ifdef ENABLE_CUBE_COLLISION
    float xDistance = fabs(output.x) - CUBE_SIZE;
    float yDistance = fabs(output.y) - CUBE_SIZE;
    float zDistance = fabs(output.z) - CUBE_SIZE;
    if (xDistance < 0.0f && yDistance < 0.0f && zDistance < 0.0f)
    {
        if (xDistance > yDistance && xDistance > zDistance)
        {
            if (output.x > 0.0f)
                output.x = CUBE_SIZE;
            else
                output.x = -CUBE_SIZE;
        }
        else if (yDistance > zDistance)
        {
            if (output.y > 0.0f)
                output.y = CUBE_SIZE;
            else
                output.y = -CUBE_SIZE;
        }
        else
        {
            if (output.z > 0.0f)
                output.z = CUBE_SIZE;
            else
                output.z = -CUBE_SIZE;
        }
    }
#endif
    
#ifdef ENABLE_PLANE_COLLISION
    if (output.z <= PLANE_HEIGHT)
        output.z = PLANE_HEIGHT;
#endif
    positions[id] = output;
}

__kernel void timeStep(__global float4* old_positions,
                       __global float4* positions)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    size_t id = y * (CLOTH_SIZE) + x;
    
    if (x < 0 || y < 0 || x >= CLOTH_SIZE || y >= CLOTH_SIZE)
        return;
    
    old_positions[id] = positions[id];
}

float4 get_clamped_node(__global float4* positions, int x, int y)
{
    x = max(0, min(CLOTH_SIZE - 1, x));
    y = max(0, min(CLOTH_SIZE - 1, y));
    size_t id = y * CLOTH_SIZE + x;
    return positions[id];
}

__kernel void calculateNormals(__global float4* positions,
                               __global float4* normals)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    size_t id = y * (CLOTH_SIZE) + x;
    
    if (x < 0 || y < 0 || x >= CLOTH_SIZE || y >= CLOTH_SIZE)
        return;

	float4 output = positions[id];
	float4 right = get_clamped_node(positions, x + 1, y);
	float4 left = get_clamped_node(positions, x - 1, y);
	float4 down = get_clamped_node(positions, x, y + 1);
	float4 up = get_clamped_node(positions, x, y - 1);
    
	float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};

    if (x > 0 && y > 0)
        sum += cross(left - output, up - output);
    if (x > 0 && y < (CLOTH_SIZE - 1))
        sum += cross(down - output, left - output);
    if (x < (CLOTH_SIZE - 1) && y > (CLOTH_SIZE - 1))
        sum += cross(right - output, down - output);
    if (x < (CLOTH_SIZE - 1) && y < (CLOTH_SIZE - 1))
        sum += cross(up - output, right - output);
    
    normals[id] = sum / fast_length(sum);
}
