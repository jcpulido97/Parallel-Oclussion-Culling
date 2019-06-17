
#include "octree.h"
// #include "node.h"
// #include "cudaquery.cuh"
#include <iostream>

#include <fstream>
#include <iomanip>

#include <stdlib.h>

#include <deque>
#include <unordered_set>
#include <random>

#include <GL/glew.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <GLFW/glfw3.h>

using namespace std;

#define VERTEX_PER_OBJ 6144
#define TRIANGLES_PER_OBJ VERTEX_PER_OBJ/3

const float X_MIN=-0.5;
const float X_MAX=0.5;
const float Y_MIN=-0.5;
const float Y_MAX=0.5;
const float FRONT_PLANE_PERSPECTIVE=(X_MAX-X_MIN)/2;
const float BACK_PLANE_PERSPECTIVE=1000;
const float DEFAULT_DISTANCE=10;

float ANGLE_STEP=1;
const int AXIS_SIZE=5000;


float Observer_angle_x;
float Observer_angle_y;
float Observer_distance;
float escala = 1.0;
int width, height;
int discarded_obj = 0;
int total_obj = 0;

vec camera_pos = {0,0,0};
vec camera_dir = {0,0,-1};

int nbFrames = 0;
double lastTime = 0.0;

math::Frustum frustum;

bool occlusion = false;
bool reverse_paint = false;

void error_callback(int error, const char* description){
    fprintf(stderr, "Error%i: %s\n", error, description);
}



void clear_window()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
}


//**************************************************************************
// Funcion para definir la transformación de proyeccion
//***************************************************************************

void change_projection()
{
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  // formato(x_minimo,x_maximo, y_minimo, y_maximo,Front_plane, plano_traser)
  //  Front_plane>0  Back_plane>PlanoDelantero)
  glFrustum(X_MIN*escala,X_MAX*escala,Y_MIN,Y_MAX,FRONT_PLANE_PERSPECTIVE,BACK_PLANE_PERSPECTIVE);
}

//**************************************************************************
// Funcion para definir la transformación de vista (posicionar la camara)
//***************************************************************************

void change_observer()
{
  // posicion del observador
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0,0,-Observer_distance);
  glRotatef(Observer_angle_x,1,0,0);
  glRotatef(Observer_angle_y,0,1,0);
  glTranslatef(-camera_pos.x, -camera_pos.y, -camera_pos.z);
}

//**************************************************************************
// Funcion que dibuja los ejes utilizando la primitiva grafica de lineas
//***************************************************************************

void draw_axis()
{
  glBegin(GL_LINES);
  // eje X, color rojo
  glColor3f(1,0,0);
  glVertex3f(-AXIS_SIZE,0,0);
  glVertex3f(AXIS_SIZE,0,0);
  // eje Y, color verde
  glColor3f(0,1,0);
  glVertex3f(0,-AXIS_SIZE,0);
  glVertex3f(0,AXIS_SIZE,0);
  // eje Z, color azul
  glColor3f(0,0,1);
  glVertex3f(0,0,-AXIS_SIZE);
  glVertex3f(0,0,AXIS_SIZE);
  glEnd();
}

void paintGL()
{
  clear_window();
  change_projection();
  change_observer();
  draw_axis();
}

//*************************************************************************
//
//*************************************************************************

void resizeGL(GLFWwindow* window, int Width1, int Height1)
{
  width = Width1;
  height = Height1;
  escala = (1.0*Width1)/Height1;
  glViewport(0,0,Width1,Height1);
}

////////////////////////////////////////////////////////////////////////

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode){

    switch(key){
      case GLFW_KEY_W: camera_pos.z-=ANGLE_STEP;break;
      case GLFW_KEY_A: camera_pos.x-=ANGLE_STEP;break;
      case GLFW_KEY_S: camera_pos.z+=ANGLE_STEP;break;
      case GLFW_KEY_D: camera_pos.x+=ANGLE_STEP;break;
      case GLFW_KEY_PAGE_UP:  camera_pos.y+=ANGLE_STEP;break;
      case GLFW_KEY_PAGE_DOWN:camera_pos.y-=ANGLE_STEP;break;
      case GLFW_KEY_UP: Observer_angle_x-=ANGLE_STEP;break;
      case GLFW_KEY_LEFT: Observer_angle_y-=ANGLE_STEP;break;
      case GLFW_KEY_DOWN: Observer_angle_x+=ANGLE_STEP;break;
      case GLFW_KEY_RIGHT: Observer_angle_y+=ANGLE_STEP;break;
      case GLFW_KEY_P: if(action == GLFW_PRESS) reverse_paint=!reverse_paint;break;
      case GLFW_KEY_O:
        if(action == GLFW_PRESS){
          occlusion=!occlusion;
          ANGLE_STEP = occlusion ? 0.5 : 1;
        }
      ;break;
      case GLFW_KEY_ESCAPE:glfwSetWindowShouldClose(window, true);break;
    }
    paintGL();
}

void showFPS(GLFWwindow *pWindow)
{
    // Measure speed
     double currentTime = glfwGetTime();
     double delta = currentTime - lastTime;
     ++nbFrames;
     if ( delta >= 1.0 ){ // If last cout was more than 1 sec ago
         double fps = double(nbFrames) / delta;

         std::stringstream ss;
         ss << "TFG [" << fps << " FPS] - Total Objetos = " << total_obj << " --- Después de los descartes = " << discarded_obj;

         glfwSetWindowTitle(pWindow, ss.str().c_str());

         nbFrames = 0;
         lastTime = currentTime;
     }
}

int main(int argc,  char* argv[]){
  int size;
  if(argc == 1){
    size = 500;
  }
  else{
    size = atoi(argv[1]);
  }

  if (!glfwInit()){
    cerr << "Error al inicializar GLFW\n";
    exit(-1);
  }
  glfwSetErrorCallback(error_callback);

#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Para hacer feliz a MacOS ; Aunque no debería ser necesaria
#endif

  //Crear una ventana y su contexto OpenGL
  GLFWwindow* window; // (En el código que viene aqui, está variable es global)
  window = glfwCreateWindow( 1920, 1080, "TFG", NULL, NULL);
  if( window == NULL ){
      fprintf( stderr, "Falla al abrir una ventana GLFW. Si usted tiene una GPU Intel, está no es compatible con 3.3. Intente con la versión 2.1 de los tutoriales.\n" );
      glfwTerminate();
      return -1;
  }
  glfwMakeContextCurrent(window); // Inicializar GLEW
  glfwSetKeyCallback(window, key_callback);
  glfwSetFramebufferSizeCallback(window, resizeGL);
  // glewExperimental=true; // Se necesita en el perfil de base.
  if (glewInit() != GLEW_OK) {
      fprintf(stderr, "Falló al inicializar GLEW\n");
      return -1;
  }

  glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
  glEnable(GL_DEPTH_TEST);
  glfwSwapInterval(0);

  glfwGetFramebufferSize(window, &width, &height);
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
  glViewport(0, 0, width, height);
  escala = (1.0*width)/height;
  glFrustum(X_MIN*escala,X_MAX*escala,Y_MIN,Y_MAX,FRONT_PLANE_PERSPECTIVE,BACK_PLANE_PERSPECTIVE);

  const GLubyte* strm;

  strm = glGetString(GL_VENDOR);
  std::cerr << "Vendor: " << strm << "\n";
  strm = glGetString(GL_RENDERER);
  std::cerr << "Renderer: " << strm << "\n";
  strm = glGetString(GL_VERSION);
  std::cerr << "OpenGL Version: " << strm << "\n";

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);

  total_obj = size;
  vector<math::OBB> objs(size);
  std::vector<std::vector<math::vec>> VBO(size);
  for(auto& it : VBO)
    it.resize(VERTEX_PER_OBJ);

  auto pos = math::vec(dis(gen)*100.0, dis(gen)*100.0, dis(gen)*-100.0);
  auto sphere = math::Sphere(pos, dis(gen)*5.0);
  for(int i = 0; i < size; ++i){
    objs[i].SetFrom(sphere);
    // sphere.Triangulate(VBO[i].data(), nullptr, nullptr, VERTEX_PER_OBJ, true);
    objs[i].Triangulate(1,1,1, VBO[i].data(), nullptr, nullptr, true);
    pos = math::vec(dis(gen)*100.0, dis(gen)*100.0, dis(gen)*-100.0);
    sphere = math::Sphere(pos, dis(gen)*5.0);
  }

  // vector<math::OBB> objs(2);
  objs[0] = math::AABB({0,0, 5}, {100,100,10});         // ocluder
  objs[0].Triangulate(1,1,1, VBO[0].data(), nullptr, nullptr, true);
  sphere = math::Sphere({5,5,-50}, 1);
  objs[1].SetFrom(sphere);         // ocluder
  objs[1].Triangulate(1,1,1, VBO[1].data(), nullptr, nullptr, true);
  // objs[2].SetFrom(math::Sphere({6,0,0}, 1));         // Separated object

  // sphere = math::Sphere({6,0,0}, 1);
  // sphere.Triangulate(VBO[2].data(), nullptr, nullptr, VERTEX_PER_OBJ, true);
  // objs[2].SetFrom(math::Sphere({6,0,0}, 1));         // Separated object


  Octree oc(objs);

  math::vec camera_dir_normal = {1,1,1};
  camera_dir_normal.Normalize();
  camera_dir = {camera_dir_normal.x,camera_dir_normal.y,camera_dir_normal.z};
  math::vec camera_up = {0,1,0};
  vector<unsigned int> visible_objs;
  frustum.SetPos({camera_pos.x,camera_pos.y,camera_pos.z});
  frustum.SetFront({camera_dir.x,camera_dir.y,camera_dir.z});
  frustum.SetUp(camera_up);
  frustum.SetKind(math::FrustumSpaceGL,math::FrustumRightHanded);
  frustum.SetViewPlaneDistances(FRONT_PLANE_PERSPECTIVE, BACK_PLANE_PERSPECTIVE);
  auto octree_cut = oc.computeOcclusions(frustum, camera_pos, camera_dir, &visible_objs);

  cout << "Visible objects " << visible_objs.size() << " - Octree cut "<< octree_cut <<" - Original size " << objs.size() << endl;
  octree_cut = oc.computeOcclusions(frustum, camera_pos, camera_dir, &visible_objs);
  cout << "Visible objects " << visible_objs.size() << " - Octree cut "<< octree_cut <<" - Original size " << objs.size() << endl;
  // Ensure we can capture the escape key being pressed below
  glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

  GLfloat matrix[16];
  do{
    showFPS(window);
    paintGL();
    // glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    // glColor3f(0.75,0.75,0.75);
    // glBegin(GL_TRIANGLES);
    //   glVertex3f(-1.0f, -1.0f, 0.0f);
    //   glVertex3f( 1.0f, -1.0f, 0.0f);
    //   glVertex3f( 0.0f,  1.0f, 0.0f);
    // glEnd();
    // glColor3f(0,0,1);
    // glPointSize(4);
    // glBegin(GL_POINTS);
    //   glVertex3f(-1.0f, -1.0f, 0.0f);
    //   glVertex3f( 1.0f, -1.0f, 0.0f);
    //   glVertex3f( 0.0f,  1.0f, 0.0f);
    // glEnd();
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable( GL_BLEND );
    if(occlusion){
      glGetFloatv( GL_MODELVIEW_MATRIX, matrix );

      camera_dir_normal = {matrix[  2 ], matrix[  6 ], matrix[ 10 ]};
      camera_dir_normal.Normalize();
      camera_dir_normal*=-1;
      camera_dir = {camera_dir_normal.x,camera_dir_normal.y,camera_dir_normal.z};
      frustum.SetFront({camera_dir.x,camera_dir.y,camera_dir.z});
      visible_objs.clear();
      oc.computeOcclusions(frustum, camera_pos, camera_dir, &visible_objs);
      discarded_obj = visible_objs.size();
      // cout << "{" << camera_pos.x  << ", " << camera_pos.y << ", " << camera_pos.z << "} - ";
      // cout << "{" << camera_dir.x  << ", " << camera_dir.y << ", " << camera_dir.z << "}\n";
      // cout << "Total objects " << VBO.size()  << " - After occlusion culling " << visible_objs.size() << "\n";
      if(reverse_paint)
        std::sort(visible_objs.rbegin(), visible_objs.rend());

      for(unsigned int i= 0; i < visible_objs.size(); ++i){
        for(unsigned int face= 0; face < VBO[i].size(); face+=3){
          glPolygonMode(GL_FRONT,GL_FILL);
          // glColor4f((visible_objs[i])%255, (visible_objs[i])%255, visible_objs[i]%255, 0.85);
          glColor4f(200, 200, 200, 0.75);

          glBegin(GL_TRIANGLES);
            glVertex3f(VBO[visible_objs[i]][face].x,   VBO[visible_objs[i]][face].y,   VBO[visible_objs[i]][face].z);
            glVertex3f(VBO[visible_objs[i]][face+1].x, VBO[visible_objs[i]][face+1].y, VBO[visible_objs[i]][face+1].z);
            glVertex3f(VBO[visible_objs[i]][face+2].x, VBO[visible_objs[i]][face+2].y, VBO[visible_objs[i]][face+2].z);
          glEnd();
        }
      }
    }
    else{
      discarded_obj = size;
      if(!reverse_paint){
        for(unsigned int i= 0; i< VBO.size(); ++i){
          for(unsigned int face= 0; face < VBO[i].size(); face+=3){
            glPolygonMode(GL_FRONT,GL_FILL);
            glColor4f(200, 200, 200, 0.75);
            glBegin(GL_TRIANGLES);
              glVertex3f(VBO[i][face].x,   VBO[i][face].y,   VBO[i][face].z);
              glVertex3f(VBO[i][face+1].x, VBO[i][face+1].y, VBO[i][face+1].z);
              glVertex3f(VBO[i][face+2].x, VBO[i][face+2].y, VBO[i][face+2].z);
            glEnd();
          }
        }
      }
      else{
        for(int i=  VBO.size()-1; i>=0; --i){
          for(unsigned int face= 0; face < VBO[i].size(); face+=3){
            glPolygonMode(GL_FRONT,GL_FILL);
            glColor4f(200, 200, 200, 0.75);
            glBegin(GL_TRIANGLES);
              glVertex3f(VBO[i][face].x,   VBO[i][face].y,   VBO[i][face].z);
              glVertex3f(VBO[i][face+1].x, VBO[i][face+1].y, VBO[i][face+1].z);
              glVertex3f(VBO[i][face+2].x, VBO[i][face+2].y, VBO[i][face+2].z);
            glEnd();
          }
        }
      }
    }
    // Swap buffers
    glfwSwapBuffers(window);
    glfwPollEvents();
  } // Check if the ESC key was pressed or the window was closed
  while(glfwWindowShouldClose(window) == 0);
  glfwTerminate();
}
