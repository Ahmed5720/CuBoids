#pragma once

#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "utilityCore.hpp"
#include "glslUtility.hpp"

//====================================
// GL Stuff
//====================================

GLuint positionLocation = 0;   // Match results from glslUtility::createProgram.
GLuint velocitiesLocation = 1; // Also see attribtueLocations below.
const char *attributeLocations[] = { "Position", "Velocity" };

GLuint boidVAO = 0;
GLuint boidVBO_positions = 0;
GLuint boidVBO_velocities = 0;
GLuint boidIBO = 0;
GLuint displayImage;
GLuint program[2];
// to visualize mesh
GLuint meshVAO = 0, meshVBO = 0, meshIBO = 0, meshProgram = 0;
GLsizei meshIndexCount = 0;
GLint meshProjLoc = -1;
GLuint meshTex = 0;
GLuint meshNormalVBO;

const unsigned int PROG_BOID = 0;

const float fovy = (float) (PI / 4);
const float zNear = 0.10f;
const float zFar = 10.0f;
// LOOK-1.2: for high DPI displays, you may want to double these settings.
int width = 1280;
int height = 720;
int pointSize = 2;

// For camera controls
bool leftMousePressed = false;
bool rightMousePressed = false;
double lastX;
double lastY;
float theta = 1.22f;
float phi = -0.70f;
float zoom = 4.0f;
glm::vec3 lookAt = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 cameraPosition;

glm::mat4 projection;

std::vector<glm::vec3> originalVerts;
std::vector<glm::vec3> meshVerts;
std::vector<glm::vec3> meshNormals;
std::vector<unsigned int> mtris;

// Mesh transform
glm::vec3 meshPosition(0.0f, 0.0f, 70.0f);
glm::vec3 meshRotation(90.0f, 180.0f, 0.0f);
float meshScale = 80.0f;



static bool ui_hide = false;

//====================================
// Main
//====================================

const char *projectName;

int main(int argc, char* argv[]);

//====================================
// Main loop
//====================================
void mainLoop();
void errorCallback(int error, const char *description);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void updateCamera();
void runCUDA();

//====================================
// Setup/init Stuff
//====================================
bool init(int argc, char **argv);
void initVAO();
void initMesh(const std::vector<glm::vec3>& verts, const std::vector<glm::vec3>& normals,
    const std::vector<glm::vec2>& uvs,
    const std::vector<unsigned int>& indices,
    const std::vector<unsigned char>& texPixels, int texW, int texH, int texCh);
void initShaders(GLuint *program);
std::vector<glm::vec3> sampleMeshSurface(const std::vector<glm::vec3>& verts, std::vector<unsigned int>& indices, float spacing, int layers);
void placeMesh(std::vector<glm::vec3>& verts,
    float targetSize,
    glm::vec3 eulerDeg,    // rotation in degrees, applied X then Y then Z
    glm::vec3 position);
bool loadGltf(const char* path,
    std::vector<glm::vec3>& verts, std::vector<glm::vec3>& normals,
    std::vector<glm::vec2>& uvs,
    std::vector<unsigned int>& indices,
    std::vector<unsigned char>& texPixels, int& texW, int& texH, int& texCh);
void updateMeshTransform();
void  uploadMeshVertices();
void restartSimulation();
