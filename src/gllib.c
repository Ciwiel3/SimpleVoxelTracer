#include <glad/glad.h>
#include <stdlib.h>
#include "gllib.h"
#include "cplog.h"
#define STB_INCLUDE_IMPLEMENTATION
#define STB_INCLUDE_LINE_NONE
#include "stb_include.h"
#include "cpmath.h"

static u32 makeShader(const char* path, GLenum shaderType);

u32 gllib_makePipeline(const char *vertPath, const char *fragPath)
{
    u32 vertShader = makeShader(vertPath, GL_VERTEX_SHADER);
    u32 fragShader = makeShader(fragPath, GL_FRAGMENT_SHADER);

    u32 shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertShader);
    glAttachShader(shaderProgram, fragShader);
    glLinkProgram(shaderProgram);

    GLint isLinked = 0;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &isLinked);
    if (isLinked == GL_FALSE)
    {
    	GLint maxLength = 0;
    	glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &maxLength);
    	char errorLog[maxLength];
	    glGetProgramInfoLog(shaderProgram, maxLength, &maxLength, errorLog);
    	LOG_ERROR("ERROR COMPILING SHADER: %s", errorLog);
    	glDeleteProgram(shaderProgram);
        exit(-1);
    }

    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    return shaderProgram;
}

u32 gllib_makeCompute(const char *shaderPath)
{
    u32 shader = makeShader(shaderPath, GL_COMPUTE_SHADER);

    u32 shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, shader);
    glLinkProgram(shaderProgram);

    GLint isLinked = 0;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &isLinked);
    if (isLinked == GL_FALSE)
    {
        GLint maxLength = 0;
        glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &maxLength);
        char errorLog[maxLength];
        glGetProgramInfoLog(shaderProgram, maxLength, &maxLength, errorLog);
        LOG_ERROR("ERROR COMPILING SHADER: %s", errorLog);
        glDeleteProgram(shaderProgram);
        exit(-1);
    }

    glDeleteShader(shader);

    return shaderProgram;
}

Texture gllib_makeDefaultTexture(u32 width, u32 height, u32 glInternalFormat, u32 glFilter)
{
    Texture out;
    out.internalFormat = glInternalFormat;
    glCreateTextures(GL_TEXTURE_2D, 1, &out.handle);
    glTextureParameteri(out.handle, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(out.handle, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if (glFilter)
    {
        glTextureParameteri(out.handle, GL_TEXTURE_MAG_FILTER, glFilter);
        glTextureParameteri(out.handle, GL_TEXTURE_MIN_FILTER, glFilter);
    }
    glTextureStorage2D(out.handle, 1, glInternalFormat, width, height);
    return out;
}

void gllib_destroyTexture(Texture *texture)
{
    glDeleteTextures(1, &texture->handle);
}

void gllib_bindTexture(const Texture *texture, u32 idx, u32 glUsage)
{
    glBindImageTexture(idx, texture->handle, 0, GL_FALSE, 0, glUsage, texture->internalFormat);
}

static u32 makeShader(const char* path, GLenum shaderType)
{
    char error[256];
    char* code = stb_include_file((char*) path, (char*) "", (char*) "res/shaders/inc", error);
    if(!code)
    {
        LOG_ERROR("Error Parsing Shader: %s", error);
        exit(-1);
    }

    u32 shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, (const GLchar* const*) &code, NULL);
    glCompileShader(shader);

    GLint isCompiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);
    if(isCompiled == GL_FALSE)
    {
    	GLint maxLength = 0;
    	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);
    	char errorLog[maxLength];
    	glGetShaderInfoLog(shader, maxLength, &maxLength, errorLog);
        LOG_ERROR("ERROR COMPILING SHADER: %s", errorLog);
    	glDeleteShader(shader); // Don't leak the shader.
        exit(-1);
    }

    free(code);
    return shader;
}
