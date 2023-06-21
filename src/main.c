#include <stdio.h>
#include <time.h>
#include "graphics.h"
#include "terrain.h"
#include "cplog.h"
#include "GLFW/glfw3.h"
#include "cptime.h"

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

static void updateCamera(float dTimeS);

static vec3 camPos;
static vec3 forward;

int main(void)
{
    graphics_init();

    Terrain terrain;

    // generate terrain
    u32 start = clock();
    terrain_init(&terrain, 1024, 256);
    u32 stop = clock();

    // calc memory footprint
    u32 poolSize = terrain.chunkPool.size * terrain.chunkPool.unitSize + terrain.chunkBitmaskPool.size * terrain.chunkBitmaskPool.unitSize;
    u32 terrainByteSize = poolSize + terrain.chunkCount * 4 + terrain.chunkCount / 8;

    LOG_INFO("Generation took: %ums", ((stop - start)));
    LOG_INFO("Memory: %u bytes", terrainByteSize);
    LOG_INFO("Chunks: %u", terrain.chunkPool.size);

    glfwSetKeyCallback(glfwGetCurrentContext(), key_callback);

    camPos = (vec3) {terrain.width / 2, terrain.height / 2, 10};
    forward = normalize(((vec3) {0, -2, 3}));

    u32 time = uclock();
    u32 frameTime = 1;
    u32 accum = 0;
    u32 count = 0;
    while (!glfwWindowShouldClose(glfwGetCurrentContext())) {

        glfwPollEvents();

        updateCamera(frameTime / 1000000.0f);

        graphics_drawFrame(&terrain, camPos, forward);

        // frame time
        frameTime = uclock() - time;
        time = uclock();
        accum += frameTime;
        count++;
        if (accum >= 1000000)
        {
            LOG_INFO("Frame Time: %.2fms", (accum / (float) count / 1000.0f));
            accum = 0;
            count = 0;
        }
    }

    // clean up
    terrain_destroy(&terrain);
    graphics_destroy();

    LOG_INFO("Press any Enter to exit.")
    getchar();

    return 0;
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_F5 && action == GLFW_PRESS)
        graphics_reloadShaders();

    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        camPos.y += 2;

    if (key == GLFW_KEY_LEFT_CONTROL && action == GLFW_PRESS)
        camPos.y -= 2;
}

static void updateCamera(float dTimeS)
{
    GLFWwindow* window = glfwGetCurrentContext();

    float speed = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ? 50 : 20;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        camPos = add(camPos, mul(forward, dTimeS * speed));
    }

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        camPos = add(camPos, mul(forward, -dTimeS * speed));
    }

    vec3 right = cross(forward, ((vec3) {0, 1, 0}));

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        camPos = add(camPos, mul(right, -dTimeS * speed));
    }

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        camPos = add(camPos, mul(right, dTimeS * speed));
    }

    static double oldPosX = 0.0;
    static double oldPosY = 0.0;

    double posX = 0.0;
    double posY = 0.0;

    glfwGetCursorPos(window, &posX, &posY);

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT))
    {
        double dX = posX - oldPosX;
        double dY = posY - oldPosY;

        if (dX != 0) {
            forward = normalize(add(forward, mul(right, (float) -dX / 1000.0f)));
        }

        if (dY != 0) {
            forward = normalize(add(forward, mul(((vec3) {0, 1, 0}), (float) dY / 1000.0f)));
        }
    }

    oldPosX = posX;
    oldPosY = posY;
}
