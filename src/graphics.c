#include <time.h>
#include "graphics.h"
#include "cpmath.h"
#include "cplog.h"
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "gllib.h"
#include "cptime.h"

static const int DEFAULT_WINDOW_WIDTH = 1280;
static const int DEFAULT_WINDOW_HEIGHT = 720;

static void createWindowAndContext(void);
static void freeWindowAndContext(void);

static void createSizeAwareResources(void);
static void freeSizeAwareResources(void);

static void createPermanentResources(void);
static void freePermanentResources(void);

static void createWorldResources(void);
static void freeWorldResources(void);

static void loadShaders(void);
static void freeShaders(void);

// callback for opengl
static void APIENTRY glDebugOutput(GLenum source,
                            GLenum type,
                            unsigned int id,
                            GLenum severity,
                            GLsizei length,
                            const char *message,
                            const void *userParam);

static void framebuffer_size_callback(GLFWwindow* window, int width, int height);

// ##### STATE ####

static GLFWwindow* window;
static u32 resX;
static u32 resY;

static bool renderingResourcesCreated = false;
static bool shadersLoaded = false;

static u32 terrainChunkArraySSBO;
static u32 terrainPoolSSBO;
static u32 terrainBitPoolSSBO;

static u32 currentPoolBufferSize = 0;

static u32 fbComputeTarget;

static u32 shaderTerrainInitial;
static u32 shaderDFGenPrepare;
static u32 shaderDFGenX;
static u32 shaderDFGenY;
static u32 shaderDFGenZ;

static Texture texTerrainInitial;

// ################

void graphics_init(void)
{
    createWindowAndContext();
    createWorldResources();
    createPermanentResources();
    createSizeAwareResources();
    loadShaders();
}

void graphics_destroy(void)
{
    freeShaders();
    freeSizeAwareResources();
    freePermanentResources();
    freeWorldResources();
    freeWindowAndContext();
}

void graphics_drawFrame(Terrain *terrain, vec3 camPos, vec3 forward)
{
    mat4 viewMat = worldToCamMatrix(camPos,
                              forward,
                              (vec3) {0, 1, 0});

    mat4 projMat = perspectiveProjectionMatrix(radians(70.0f), resX / (float) resY, 0.01, 1000);

    if (terrain->dirty)
    {
        // TODO: keep track of dirty chunks instead of flushing the entire buffer
        // update / create top level chunk array SSBOs
        if (currentPoolBufferSize == 0)
        {
            glNamedBufferData(terrainChunkArraySSBO, terrain->chunkCount * sizeof(u32), terrain->topLevelArray, GL_STATIC_DRAW);
        }
        else
        {
            glNamedBufferSubData(terrainChunkArraySSBO, 0, terrain->chunkCount * sizeof(u32), terrain->topLevelArray);
        }

        // update / create pool SSBOs
        if (terrain->chunkPool.maxSize != currentPoolBufferSize)
        {
            glNamedBufferData(terrainPoolSSBO, terrain->chunkPool.maxSize * terrain->chunkPool.unitSize, terrain->chunkPool.memory, GL_STATIC_DRAW);
            glNamedBufferData(terrainBitPoolSSBO, terrain->chunkBitmaskPool.maxSize * terrain->chunkBitmaskPool.unitSize, terrain->chunkBitmaskPool.memory, GL_STATIC_DRAW);
        }
        else
        {
            glNamedBufferSubData(terrainPoolSSBO, 0, terrain->chunkPool.maxSize * terrain->chunkPool.unitSize, terrain->chunkPool.memory);
            glNamedBufferSubData(terrainBitPoolSSBO, 0, terrain->chunkBitmaskPool.maxSize * terrain->chunkBitmaskPool.unitSize, terrain->chunkBitmaskPool.memory);
        }
        terrain->dirty = false;

        glFinish();
        u32 start = uclock();
        // generate distance field
        // prepare pass (set all empty chunk DF values to highest)
        glUseProgram(shaderDFGenPrepare);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, terrainChunkArraySSBO);
        glUniform3ui(0, terrain->width, terrain->height, terrain->width);
        glDispatchCompute(terrain->width / 64, terrain->width / 64, 1);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // Z Pass - spread in 2 passes along Z and -Z
        glUseProgram(shaderDFGenZ);
        glUniform3ui(0, terrain->width, terrain->height, terrain->width);
        glDispatchCompute(terrain->width / 64, terrain->height / 64, 1);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // X Pass - spread in 2 passes along X and -X
        glUseProgram(shaderDFGenX);
        glUniform3ui(0, terrain->width, terrain->height, terrain->width);
        glDispatchCompute(terrain->height / 64, terrain->width / 64, 1);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // Y Pass - spread in 2 passes along Y and -Y
        glUseProgram(shaderDFGenY);
        glUniform3ui(0, terrain->width, terrain->height, terrain->width);
        glDispatchCompute(terrain->width / 64, terrain->width / 64, 1);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        glFinish();
        u32 finish = uclock();
        LOG_INFO("Building DF for %u x %u x %u nodes took: %.02fms", terrain->width / 8, terrain->height / 8, terrain->width / 8, (finish - start) / 1000.0f);
    }

    // render terrain (initial ray tracing)
    glUseProgram(shaderTerrainInitial);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, terrainChunkArraySSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, terrainPoolSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, terrainBitPoolSSBO);

    gllib_bindTexture(&texTerrainInitial, 0, GL_WRITE_ONLY);

    glUniform2ui(glGetUniformLocation(shaderTerrainInitial, "screenSize"), resX, resY);
    glUniform3ui(glGetUniformLocation(shaderTerrainInitial, "terrainSize"), terrain->width, terrain->height, terrain->width);
    glUniform3f(glGetUniformLocation(shaderTerrainInitial, "camPos"), camPos.x, camPos.y, camPos.z);

    glUniformMatrix4fv(glGetUniformLocation(shaderTerrainInitial, "viewMat"), 1, GL_FALSE, viewMat.arr);
    glUniformMatrix4fv(glGetUniformLocation(shaderTerrainInitial, "projMat"), 1, GL_FALSE, projMat.arr);

    glDispatchCompute(ceilf(resX / 8.0f), ceilf(resY / 8.0f), 1);

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // blit compute output to screen
    glBlitNamedFramebuffer(fbComputeTarget, 0,
                           0, 0, resX, resY,
                           0, 0, resX, resY,
                           GL_COLOR_BUFFER_BIT, GL_NEAREST);

    glfwSwapBuffers(window);
}

void graphics_reloadShaders(void)
{
    loadShaders();
}

uvec2 graphics_getRes(void)
{
    return (uvec2) {resX, resY};
}

static void createWindowAndContext(void)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint( GLFW_DOUBLEBUFFER, GLFW_TRUE);
//    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);

    resX = DEFAULT_WINDOW_WIDTH;
    resY = DEFAULT_WINDOW_HEIGHT;

    // create new window
    window = glfwCreateWindow(resX, resY, "Simple Voxel Renderer", NULL, NULL);

    if (window == NULL)
    {
        LOG_ERROR("Failed to create GLFW window");
        glfwTerminate();
        exit(-1);
    }

    glfwMakeContextCurrent(window);

    // vsync
    glfwSwapInterval(false);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        LOG_ERROR("Failed to initialize GLAD");
        exit(-1);
    }

    int flags; glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
    if (flags & GL_CONTEXT_FLAG_DEBUG_BIT)
    {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(glDebugOutput, NULL);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
    }

    glViewport(0, 0, resX, resY);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
}

static void freeWindowAndContext(void)
{
    glfwDestroyWindow(glfwGetCurrentContext());
}

static void createWorldResources(void)
{
    glCreateBuffers(1, &terrainChunkArraySSBO);
    glCreateBuffers(1, &terrainPoolSSBO);
    glCreateBuffers(1, &terrainBitPoolSSBO);
}

static void freeWorldResources(void)
{
    glDeleteBuffers(1, &terrainChunkArraySSBO);
    glDeleteBuffers(1, &terrainPoolSSBO);
    glDeleteBuffers(1, &terrainBitPoolSSBO);
}

static void createSizeAwareResources(void)
{
    if (renderingResourcesCreated)
        freeSizeAwareResources();

    texTerrainInitial = gllib_makeDefaultTexture(resX, resY, GL_RGBA8, GL_NEAREST);

    // bind final compute output to framebuffer, so it can be blit to screen
    glNamedFramebufferTexture(fbComputeTarget,  GL_COLOR_ATTACHMENT0, texTerrainInitial.handle, 0);

    renderingResourcesCreated = true;
}

static void freeSizeAwareResources(void)
{
    if (!renderingResourcesCreated)
        return;

    gllib_destroyTexture(&texTerrainInitial);

    renderingResourcesCreated = false;
}

static void createPermanentResources(void)
{
    glCreateFramebuffers(1, &fbComputeTarget);
}

static void freePermanentResources(void)
{
    glDeleteFramebuffers(1, &fbComputeTarget);
}

static void loadShaders(void)
{
    if (shadersLoaded)
        freeShaders();

    shaderTerrainInitial = gllib_makeCompute("res/shaders/compute/initial.glsl");
    shaderDFGenPrepare = gllib_makeCompute("res/shaders/compute/dfGenPrepare.glsl");
    shaderDFGenX = gllib_makeCompute("res/shaders/compute/dfGenXPass.glsl");
    shaderDFGenY = gllib_makeCompute("res/shaders/compute/dfGenYPass.glsl");
    shaderDFGenZ = gllib_makeCompute("res/shaders/compute/dfGenZPass.glsl");

    shadersLoaded = true;
}

static void freeShaders(void)
{
    if (!shadersLoaded)
        return;

    glDeleteProgram(shaderTerrainInitial);
    glDeleteProgram(shaderDFGenPrepare);
    glDeleteProgram(shaderDFGenX);
    glDeleteProgram(shaderDFGenY);
    glDeleteProgram(shaderDFGenZ);

    shadersLoaded = false;
}

static void framebuffer_size_callback(GLFWwindow* w, int width, int height)
{
    glViewport(0, 0, width, height);
    resX = max(1, width);
    resY = max(1, height);

    createSizeAwareResources();
}

static void APIENTRY glDebugOutput(GLenum source,
                            GLenum type,
                            unsigned int id,
                            GLenum severity,
                            GLsizei length,
                            const char *message,
                            const void *userParam)
{
    // ignore non-significant error/warning codes
    if(id == 131169 || id == 131185 || id == 131218 || id == 131204 || id == 131188) return;

    LOG_ERROR("Debug message (%u): %s", id, message);

    switch (source)
    {
        case GL_DEBUG_SOURCE_API:             LOG_ERROR("Source: API"); break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   LOG_ERROR("Source: Window System"); break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: LOG_ERROR("Source: Shader Compiler"); break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     LOG_ERROR("Source: Third Party"); break;
        case GL_DEBUG_SOURCE_APPLICATION:     LOG_ERROR("Source: Application"); break;
        case GL_DEBUG_SOURCE_OTHER:           LOG_ERROR("Source: Other"); break;
    };

    switch (type)
    {
        case GL_DEBUG_TYPE_ERROR:               LOG_ERROR("Type: Error"); break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: LOG_ERROR("Type: Deprecated Behaviour"); break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  LOG_ERROR("Type: Undefined Behaviour"); break;
        case GL_DEBUG_TYPE_PORTABILITY:         LOG_ERROR("Type: Portability"); break;
        case GL_DEBUG_TYPE_PERFORMANCE:         LOG_ERROR("Type: Performance"); break;
        case GL_DEBUG_TYPE_MARKER:              LOG_ERROR("Type: Marker"); break;
        case GL_DEBUG_TYPE_PUSH_GROUP:          LOG_ERROR("Type: Push Group"); break;
        case GL_DEBUG_TYPE_POP_GROUP:           LOG_ERROR("Type: Pop Group"); break;
        case GL_DEBUG_TYPE_OTHER:               LOG_ERROR("Type: Other"); break;
    };

    switch (severity)
    {
        case GL_DEBUG_SEVERITY_HIGH:         LOG_ERROR("Severity: high"); break;
        case GL_DEBUG_SEVERITY_MEDIUM:       LOG_ERROR("Severity: medium"); break;
        case GL_DEBUG_SEVERITY_LOW:          LOG_ERROR("Severity: low"); break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: LOG_ERROR("Severity: notification"); break;
    };
}
