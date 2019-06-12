
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

#include <GLFW/glfw3.h>

using namespace std;

void error_callback(int error, const char* description){
    fprintf(stderr, "Error: %s\n", description);
}


int main(int argc,  char* argv[]){
  int size;
  if(argc == 1){
    size = 100;
  }
  else{
    size = atoi(argv[1]);
  }

  int times = 3;
  while(--times > 0){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    vector<math::OBB> objs(size);
    auto pos = math::vec();
    for(int i = 0; i < size; ++i){
      pos = math::vec(dis(gen)*100.0, dis(gen)*100.0, dis(gen)*100.0);
      objs[i].SetFrom(math::Sphere(pos, dis(gen)*5.0));
    }


    // using namespace std::chrono;
    // auto t1 = high_resolution_clock::now();
    //
    Octree oc(objs);

    //
    // auto t2 = std::chrono::high_resolution_clock::now();
    // auto time_span = duration_cast<duration<double>>(t2 - t1);
    //
    // std::cout << "Time to create tree = " << (time_span.count()*1000) << " ms\n";
    //
    // cout << "Max level  = " << oc.getTotalLevels() << endl;
    //
    // unsigned int nodes_per_level = 0;
    // cout << "Iteracion por punteros de niveles\n";
    // int level = 0;
    // auto levels = oc.getlevel_left_to_right();
    // for(unsigned int i = 0 ; i < oc.getTotalLevels(); ++i){
    //   cout << "Level_" << level++ << " = ";
    //   auto it2 = levels[i];
    //   while(it2 != nullptr){
    //     ++nodes_per_level;
    //     cout  << " {";
    //
    //     for(auto& path : it2->getPath())
    //       cout << path << ",";
    //
    //     cout << "} " << it2->getID() << " - ";
    //
    //     it2 = it2->getRightSibling();
    //   }
    //   cout << "nullptr";
    //   cout << endl << nodes_per_level << " nodes ------------------------" <<endl;
    //   nodes_per_level = 0;
    // }

    // for(int i = 0; i < objs.size(); ++i){
    //   cout <<  objs[i] << "\n";
    // }



    vec camera_pos = {0,0,0};
    vec camera_dir;
    math::vec camera_dir_normal = {1,1,1};
    camera_dir_normal.Normalize();
    camera_dir = {camera_dir_normal.x,camera_dir_normal.y,camera_dir_normal.z};
    math::vec camera_up = {0,1,0};
    vector<unsigned int> visible_objs;
    math::Frustum frustum;
    frustum.SetPos({camera_pos.x,camera_pos.y,camera_pos.z});
    frustum.SetFront({camera_dir.x,camera_dir.y,camera_dir.z});
    frustum.SetUp(camera_up);
    frustum.SetKind(math::FrustumSpaceGL,math::FrustumRightHanded);
    frustum.SetViewPlaneDistances(1.0, 10000.0);
    frustum.SetPerspective(10.0, 10.0);
    auto octree_cut = oc.computeOcclusions(frustum, camera_pos, camera_dir, &visible_objs);
    cout << "Visible objects " << visible_objs.size() << " - Octree cut "<< octree_cut <<" - Original size " << objs.size() << endl;
    octree_cut = oc.computeOcclusions(frustum, camera_pos, camera_dir, &visible_objs);
    cout << "Visible objects " << visible_objs.size() << " - Octree cut "<< octree_cut <<" - Original size " << objs.size() << endl;
    // cout << "Visible objects = {";
    // for(auto it : visible_objs){
    //   cout << it << ",";
    // }
    // cout << "}" << endl;
    // for(auto it : visible_objs){
    //   cout << objs[it] << "\n";
    // }


    // if (!glfwInit()){
    //     // Initialization failed
    // }
    // glfwSetErrorCallback(error_callback);
    // GLFWwindow* window = glfwCreateWindow(640, 480, "My Title", NULL, NULL);
    // if (!window)
    // {
    //     // Window or OpenGL context creation failed
    // }
    // glfwMakeContextCurrent(window);
    // while (!glfwWindowShouldClose(window))
    // {
    //     // Keep running
    // }

  }
}
