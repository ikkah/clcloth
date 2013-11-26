#include <iostream>
#include <vector>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <string>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <OpenCL/cl.h>
#include <OpenGL/OpenGL.h>
#include <glut/glut.h>
#elif WIN32
#include <Windows.h>
#include <CL/opencl.h>
#include <GL/gl.h>
#include "glut/glut.h"
#pragma comment (lib, "opencl.lib")
#else
#include <CL/opencl.h>
#include <GL/gl.h>
#include <GL/glut.h>
#endif

#include "config.h"

class Cloth;

class ClothSim
{
public:
    friend class Cloth;
    
    ClothSim();
    ~ClothSim();
    
    void init();
    
private:
    void uninit();
    
    cl_program build(const std::string& filename) const;
    std::string readLines(const std::string& filename) const;
    
    cl_context context;
    std::vector<cl_device_id> devices;
    cl_program program;
    cl_command_queue commandQueue;
    cl_kernel advanceKernel;
    cl_kernel constrainEvenKernel;
    cl_kernel constrainOddKernel;
    cl_kernel stepKernel;
    cl_kernel normalsKernel;
};

ClothSim::ClothSim()
    : context(0)
    , program(0)
    , commandQueue(0)
    , advanceKernel(0)
    , constrainEvenKernel(0)
    , constrainOddKernel(0)
    , stepKernel(0)
    , normalsKernel(0)
{
}

ClothSim::~ClothSim()
{
    uninit();
}

void ClothSim::init()
{
    cl_int error = 0;
    cl_platform_id platform;
    cl_uint platform_amount;
    
    error = clGetPlatformIDs(1, &platform, &platform_amount);
    assert(!error);
    
    cl_uint devices_amount = 0;
    error = clGetDeviceIDs(platform, DEVICE_TYPE, 0, NULL, &devices_amount);
    assert(!error);
    devices.resize(devices_amount);
    error = clGetDeviceIDs(platform, DEVICE_TYPE, devices_amount, &devices[0], 0);
    assert(!error);
    
    context = clCreateContext(0, devices_amount, &devices[0], NULL, NULL, &error);
    assert(!error);
    
    commandQueue = clCreateCommandQueue(context, devices[0], NULL, &error);
    
    program = build("kernel.cl");
    
    advanceKernel = clCreateKernel(program, "advance", &error);
    constrainEvenKernel = clCreateKernel(program, "constrain", &error);
    constrainOddKernel = clCreateKernel(program, "constrain", &error);
    stepKernel = clCreateKernel(program, "timeStep", &error);
    normalsKernel = clCreateKernel(program, "calculateNormals", &error);
}

void ClothSim::uninit()
{
    clReleaseKernel(normalsKernel);
    clReleaseKernel(stepKernel);
    clReleaseKernel(constrainOddKernel);
    clReleaseKernel(constrainEvenKernel);
    clReleaseKernel(advanceKernel);
    clReleaseCommandQueue(commandQueue);
    clReleaseProgram(program);
    clReleaseContext(context);
}

cl_program ClothSim::build(const std::string& filename) const
{
    cl_program program;
    
    std::string lines = readLines(filename);
    
    cl_int error = 0;
    size_t size = cl_uint(lines.size());
    const char* start = &lines[0];
    program = clCreateProgramWithSource(context, 1, (const char**)&start, (const size_t*)&size, &error);
    
    const char *options = "-cl-denorms-are-zero -cl-strict-aliasing -cl-fast-relaxed-math -cl-mad-enable -cl-no-signed-zeros";
    error = clBuildProgram(program, 1, &devices[0], options, NULL, NULL);
    assert(!error);
    
    return program;
}

std::string ClothSim::readLines(const std::string& filename) const
{
    std::ifstream file(filename.c_str());
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

class Cloth
{
public:
    Cloth(ClothSim& sim);
    ~Cloth();
    
    void init();
    
    void reset();
    
    void step();
    void transfer();
    
    cl_float4* getVertices() { return &result[0]; }
    cl_float4* getNormals() { return &normalsResult[0]; }
    
private:
    void uninit();
    
    ClothSim& sim;
    
    std::vector<cl_float4> result;
    std::vector<cl_float4> normalsResult;
    cl_mem oldPositions;
    cl_mem positions;
    cl_mem newPositions;
    cl_mem normals;
};

Cloth::Cloth(ClothSim& sim)
    : sim(sim)
{
}

Cloth::~Cloth()
{
}

void Cloth::reset()
{
    uninit();
    init();
    step();
    transfer();
}

void Cloth::init()
{
    size_t size = CLOTH_SIZE;
    
    result.resize(size * size);
    normalsResult.resize(size * size);
    for (std::size_t i = 0; i != result.size(); ++i)
    {
        result[i].s[0] = float(i % size) / size * CLOTH_SCALE - 0.5f * CLOTH_SCALE + CLOTH_START_X;
        result[i].s[1] = float(i / size) / size * CLOTH_SCALE - 0.5f * CLOTH_SCALE + CLOTH_START_Y;
        result[i].s[2] = CLOTH_START_Z;
        result[i].s[3] = 1.0f;
    }
    
    cl_int error = 0;
    size_t verticesSize = size * size * sizeof(cl_float4);
    oldPositions = clCreateBuffer(sim.context, CL_MEM_COPY_HOST_PTR, verticesSize, &result[0], &error);
    assert(!error);
    positions = clCreateBuffer(sim.context, CL_MEM_COPY_HOST_PTR, verticesSize, &result[0], &error);
    assert(!error);
    newPositions = clCreateBuffer(sim.context, CL_MEM_WRITE_ONLY, verticesSize, NULL, &error);
    assert(!error);
    normals = clCreateBuffer(sim.context, CL_MEM_WRITE_ONLY, verticesSize, NULL, &error);
    assert(!error);
    
    error = clSetKernelArg(sim.advanceKernel, 0, sizeof(cl_mem), &oldPositions);
    assert(!error);
    error = clSetKernelArg(sim.advanceKernel, 1, sizeof(cl_mem), &positions);
    assert(!error);
    error = clSetKernelArg(sim.advanceKernel, 2, sizeof(cl_mem), &newPositions);
    assert(!error);
    
    error = clSetKernelArg(sim.constrainEvenKernel, 0, sizeof(cl_mem), &newPositions);
    assert(!error);
    error = clSetKernelArg(sim.constrainEvenKernel, 1, sizeof(cl_mem), &positions);
    assert(!error);
    error = clSetKernelArg(sim.constrainEvenKernel, 2, sizeof(cl_float4) * TEMP_SIZE * TEMP_SIZE, NULL);
    assert(!error);
    
    error = clSetKernelArg(sim.constrainOddKernel, 0, sizeof(cl_mem), &positions);
    assert(!error);
    error = clSetKernelArg(sim.constrainOddKernel, 1, sizeof(cl_mem), &newPositions);
    assert(!error);
    error = clSetKernelArg(sim.constrainOddKernel, 2, sizeof(cl_float4) * TEMP_SIZE * TEMP_SIZE, NULL);
    assert(!error);
    
    error = clSetKernelArg(sim.stepKernel, 0, sizeof(cl_mem), &oldPositions);
    assert(!error);
    error = clSetKernelArg(sim.stepKernel, 1, sizeof(cl_mem), &positions);
    assert(!error);
    
    error = clSetKernelArg(sim.normalsKernel, 0, sizeof(cl_mem), &positions);
    assert(!error);
    error = clSetKernelArg(sim.normalsKernel, 1, sizeof(cl_mem), &normals);
    assert(!error);
}

void Cloth::uninit()
{
    result.clear();
    clReleaseMemObject(oldPositions);
    clReleaseMemObject(positions);
    clReleaseMemObject(newPositions);
    clReleaseMemObject(normals);
}

void Cloth::step()
{
    cl_int error = 0;
    size_t dimensions[] = {CLOTH_SIZE, CLOTH_SIZE};
    size_t groupSizes[] = {BLOCK_SIZE, BLOCK_SIZE};
    error = clEnqueueNDRangeKernel(sim.commandQueue, sim.advanceKernel, 2, NULL, dimensions, groupSizes, 0, NULL, NULL);
    assert(!error);
    error = clEnqueueNDRangeKernel(sim.commandQueue, sim.stepKernel, 2, NULL, dimensions, groupSizes, 0, NULL, NULL);
    assert(!error);
    
    assert((SOLVER_ITERATIONS % 2) == 1);
    for (int i = 0; i != SOLVER_ITERATIONS; ++i)
    {
        bool even = (i % 2) == 0;
        cl_kernel& kernel = even ? sim.constrainEvenKernel : sim.constrainOddKernel;
        error = clEnqueueNDRangeKernel(sim.commandQueue, kernel, 2, NULL, dimensions, groupSizes, 0, NULL, NULL);
        assert(!error);
    }
    error = clEnqueueNDRangeKernel(sim.commandQueue, sim.normalsKernel, 2, NULL, dimensions, groupSizes, 0, NULL, NULL);
    assert(!error);
    
    error = clFinish(sim.commandQueue);
    assert(!error);
}

void Cloth::transfer()
{
    cl_int error = 0;
    size_t verticesSize = CLOTH_SIZE * CLOTH_SIZE * sizeof(cl_float4);
    
    error = clEnqueueReadBuffer(sim.commandQueue, positions, CL_FALSE, 0, verticesSize, (void*)&result[0], 0, NULL, NULL);
    assert(!error);
    error = clEnqueueReadBuffer(sim.commandQueue, normals, CL_FALSE, 0, verticesSize, (void*)&normalsResult[0], 0, NULL, NULL);
    assert(!error);
    
    error = clFinish(sim.commandQueue);
    assert(!error);
}

class ClothRenderer
{
public:
    ClothRenderer(Cloth& cloth);
    ~ClothRenderer();
    
    void init(int argc, char** argv);
    void loop();
    
private:
    void initGLUT(int argc, char** argv);
    void initGL();
    void uninit();
    static void keyboardFunc(unsigned char key, int x, int y);
    static void reshapeFunc(int width, int height);
    static void displayFunc();
    void display();
    void runPhysicsUpdate();
    void render();
    void moveCamera();
    void renderCollisions();
    void renderCloth();
    void renderClothNormals();
    void renderDebugInfo();
    void renderString(const std::string& s, int xPos, int yPos, float color);
    std::string makeDebugInfoString() const;
    
    static ClothRenderer* self;
    Cloth& cloth;
    
    int lastUpdateTime;
    int lastPhysicsUpdateDuration;
    int lastTransferUpdateDuration;
    int lastRenderUpdateDuration;
    
    bool paused;
    bool singleStep;
    
    bool showNormals;
    
    cl_float4 cameraOffsetPosition;
    
    GLUquadric* quadric;
};

ClothRenderer::ClothRenderer(Cloth& cloth)
    : cloth(cloth)
    , lastUpdateTime(0)
    , lastPhysicsUpdateDuration(0)
    , lastTransferUpdateDuration(0)
    , lastRenderUpdateDuration(0)
    , paused(false)
    , singleStep(false)
    , showNormals(false)
    , quadric(NULL)
{
    for (int i = 0; i != 4; ++i)
		cameraOffsetPosition.s[i] = 0.0f;
    self = this;
}

ClothRenderer::~ClothRenderer()
{
    uninit();
}

void ClothRenderer::init(int argc, char** argv)
{
    initGLUT(argc, argv);
    initGL();
}

void ClothRenderer::initGLUT(int argc, char** argv)
{
    glutInit(&argc, argv);
    
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA);
    glutInitWindowSize(640, 480);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("OpenCL Cloth Simulation");
    
    glutReshapeFunc(reshapeFunc);
    glutDisplayFunc(displayFunc);
    glutIdleFunc(displayFunc);
    glutKeyboardFunc(keyboardFunc);
}

void ClothRenderer::initGL()
{
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);
    
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    
    float ambient_light = 0.7f;
    cl_float4 light_direction;
	light_direction.s[0] = light_direction.s[1] = std::sqrt(2.0f) / 2.0f;
    light_direction.s[2] = light_direction.s[3] = 0.0f;
    float diffuse_light = 0.5f;
    
    glLightf(GL_LIGHT0, GL_AMBIENT, ambient_light);
    glLightfv(GL_LIGHT0, GL_POSITION, (float*)&light_direction);
    glLightf(GL_LIGHT0, GL_DIFFUSE, diffuse_light);
    
    quadric = gluNewQuadric();
}

void ClothRenderer::uninit()
{
    gluDeleteQuadric(quadric);
}

void ClothRenderer::loop()
{
    glutMainLoop();
}

void ClothRenderer::keyboardFunc(unsigned char key, int x, int y)
{
    if (key == '\r' || key == '\n')
        self->paused = !self->paused;
    else if (key == ' ')
        self->singleStep = true;
    else if (key == 'r')
        self->cloth.reset();
    else if (key == 'n')
        self->showNormals = !self->showNormals;
    else if (key == 'w')
        self->cameraOffsetPosition.s[1] += 1.0f;
    else if (key == 'a')
        self->cameraOffsetPosition.s[0] -= 1.0f;
    else if (key == 's')
        self->cameraOffsetPosition.s[1] -= 1.0f;
    else if (key == 'd')
        self->cameraOffsetPosition.s[0] += 1.0f;
    else if (key == 'c')
	{
		for (int i = 0; i != 4; ++i)
	        self->cameraOffsetPosition.s[i] = 0.0f;
	}
    else if (key == 'q')
        self->cameraOffsetPosition.s[2] -= 1.0f;
    else if (key == 'z')
        self->cameraOffsetPosition.s[2] += 1.0f;
}

void ClothRenderer::reshapeFunc(int width, int height)
{
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(CAMERA_FOV, width / float(height), 0.1f, 500.0f);
}

void ClothRenderer::displayFunc()
{
    self->display();
}

void ClothRenderer::display()
{
    if (!paused || singleStep)
    {
        int time = glutGet(GLUT_ELAPSED_TIME);
        int targetUpdateInterval = 1000 / TARGET_FRAME_RATE;
        bool shouldUpdate = time - lastUpdateTime > targetUpdateInterval;
        if (shouldUpdate)
        {
            lastUpdateTime = time;
            runPhysicsUpdate();
            lastPhysicsUpdateDuration = glutGet(GLUT_ELAPSED_TIME) - time;

            int transferStartTime = glutGet(GLUT_ELAPSED_TIME);
            cloth.transfer();
            lastTransferUpdateDuration = glutGet(GLUT_ELAPSED_TIME) - transferStartTime;
        }
    }
    singleStep = false;
    
    int renderStartTime = glutGet(GLUT_ELAPSED_TIME);
    render();
    lastRenderUpdateDuration = glutGet(GLUT_ELAPSED_TIME) - renderStartTime;

    glutSwapBuffers();
}

void ClothRenderer::runPhysicsUpdate()
{
    for (size_t i = 0; i != PHYSICS_TICS_PER_RENDER_FRAME; ++i)
    {
        cloth.step();
    }
}

void ClothRenderer::render()
{
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    
    moveCamera();
    renderCollisions();
    renderCloth();
    if (showNormals)
        renderClothNormals();
    renderDebugInfo();
    
    glFinish();
}

void ClothRenderer::moveCamera()
{
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    cl_float4 eye;
	eye.s[0] = CAMERA_X;
	eye.s[1] = CAMERA_Y;
	eye.s[2] = 0.0f;
	eye.s[3] = 1.0f;
	cl_float4 mid;
	mid.s[0] = 0.0f;
	mid.s[1] = 0.0f;
	mid.s[2] = 0.0f;
	mid.s[3] = 1.0f;
	cl_float4 up;
	up.s[0] = 0.0f;
	up.s[1] = 0.0f;
	up.s[2] = 1.0f;
	up.s[3] = 0.0f;
    glTranslatef(-cameraOffsetPosition.s[0], cameraOffsetPosition.s[2], cameraOffsetPosition.s[1]);
    gluLookAt(eye.s[0], eye.s[1], eye.s[2], mid.s[0], mid.s[1], mid.s[2], up.s[0], up.s[1], up.s[2]);
}

void ClothRenderer::renderCollisions()
{
    glFrontFace(GL_CCW);
    glPushMatrix();
    glColor3f(0.8f, 0.8f, 0.8f);
#ifdef ENABLE_SPHERE_COLLISION
    int slices = 64;
    glTranslatef(0.0f, 0.0f, 0.0f);
    glutSolidSphere(0.99f * SPHERE_RADIUS, slices, slices);
#endif
#ifdef ENABLE_CYLINDER_COLLISION
    int slices = 64;
    glTranslatef(0.0f, 0.0f, CYLINDER_HEIGHT - 0.5f * CYLINDER_THICKNESS);
    float radius = 0.98f * CYLINDER_RADIUS;
    float height = 0.9f * CYLINDER_THICKNESS;
    gluCylinder(quadric, radius, radius, height, slices, 1);
#endif
#ifdef ENABLE_CUBE_COLLISION
    glutSolidCube(0.97f * 2.0f * CUBE_SIZE);
#endif
    glPopMatrix();
}

void ClothRenderer::renderCloth()
{
    glFrontFace(GL_CW);
    cl_float4* vertices = self->cloth.getVertices();
    cl_float4* normals = self->cloth.getNormals();
    glBegin(GL_QUADS);
    for (int x = 0; x != CLOTH_SIZE - 1; ++x)
    {
        for (int y = 0; y != CLOTH_SIZE - 1; ++y)
        {
            // pick a color from the red/white tablecloth pattern
            bool xLine = int(x * CLOTH_SCALE / CLOTH_SIZE) % 2 == 0;
            bool yLine = int(y * CLOTH_SCALE / CLOTH_SIZE) % 2 == 0;
            if (xLine && yLine)
                glColor3f(1.0f, 1.0f, 1.0f);
            else if (xLine || yLine)
                glColor3f(1.0f, 0.5f, 0.5f);
            else
                glColor3f(1.0f, 0.0f, 0.0f);
            
            // draw the quad
            glNormal3fv((float*)&normals[x * CLOTH_SIZE + y]);
            glVertex4fv((float*)&vertices[x * CLOTH_SIZE + y]);
            glNormal3fv((float*)&normals[(x + 1) * CLOTH_SIZE + y]);
            glVertex4fv((float*)&vertices[(x + 1) * CLOTH_SIZE + y]);
            glNormal3fv((float*)&normals[(x + 1) * CLOTH_SIZE + y + 1]);
            glVertex4fv((float*)&vertices[(x + 1) * CLOTH_SIZE + y + 1]);
            glNormal3fv((float*)&normals[x * CLOTH_SIZE + y + 1]);
            glVertex4fv((float*)&vertices[x * CLOTH_SIZE + y + 1]);
        }
    }
    glEnd();
}

void ClothRenderer::renderClothNormals()
{
    cl_float4* vertices = self->cloth.getVertices();
    cl_float4* normals = self->cloth.getNormals();
    glDisable(GL_LIGHTING);
    glColor3f(0.5f, 0.5f, 1.0f);
    glBegin(GL_LINES);
    for (int x = 0; x != CLOTH_SIZE; ++x)
    {
        for (int y = 0; y != CLOTH_SIZE; ++y)
        {
            cl_float4 v = vertices[x * CLOTH_SIZE + y];
            cl_float4 n = normals[x * CLOTH_SIZE + y];
            
            const float bias = 0.05f;
            glVertex3f(v.s[0] + n.s[0] * bias, v.s[1] + n.s[1] * bias, v.s[2] + n.s[2] * bias);
            glVertex3f(v.s[0] + n.s[0], v.s[1] + n.s[1], v.s[2] + n.s[2]);
        }
    }
    glEnd();
}

void ClothRenderer::renderDebugInfo()
{
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    
    glLoadIdentity();
    gluOrtho2D(0, glutGet(GLUT_WINDOW_WIDTH), 0, glutGet(GLUT_WINDOW_HEIGHT));
    std::string s = makeDebugInfoString();
    const int xPos = 20;
    const int yPos = 20;
    
    // draw a 1px border to make text readable on white background
    renderString(s, xPos + 1, yPos, 0.0f);
    renderString(s, xPos - 1, yPos, 0.0f);
    renderString(s, xPos, yPos + 1, 0.0f);
    renderString(s, xPos, yPos - 1, 0.0f);
    
    renderString(s, xPos, yPos, 1.0f);
    
    glPopMatrix();
    
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
}

void ClothRenderer::renderString(const std::string& s, int xPos, int yPos, float color)
{
    glColor3f(color, color, color);
    glRasterPos2f(xPos, yPos);
    for (std::size_t i = 0; i != s.size(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_8_BY_13, s[i]);
    }
}

std::string ClothRenderer::makeDebugInfoString() const
{
    std::ostringstream ss;
    ss << "FPS: ";
    int totalDuration = lastPhysicsUpdateDuration + lastTransferUpdateDuration + lastRenderUpdateDuration;
    if (totalDuration > 0)
        ss << int(1000 / float(totalDuration));
    else
        ss << "N/A";
    ss << ", simulation: ";
    ss << lastPhysicsUpdateDuration;
    ss << " ms, transfer: ";
    ss << lastTransferUpdateDuration;
    ss << " ms, rendering: ";
    ss << lastRenderUpdateDuration;
    ss << " ms";
    return ss.str();
}

ClothRenderer* ClothRenderer::self = 0;

int main(int argc, char** argv)
{
	ClothSim sim;
    sim.init();
    
    Cloth cloth(sim);
    cloth.init();
    
    ClothRenderer renderer(cloth);
    renderer.init(argc, argv);
    renderer.loop();
    
    return 0;
}
