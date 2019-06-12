#ifndef OCTREE_H_
#define OCTREE_H_

#include <vector>
#include <deque>
#include <unordered_map>
#include <algorithm>
#include <limits>
//#include "LibFwd.h"
#include "node.h"

#include "../libs/MathGeoLib.h"
#include "octreeobb.h"
#include "cudaquery.cuh"

#define MAX_LEVLELS 1024

class Node;

bool checkOrderLess(Node* a, Node* b);

class Octree{
  protected:
    // Pair <group, math::OBB> Separated by levels in tree
    std::vector<bool> isOrdered;
    std::vector<Node*> level_left_to_right;
    std::unordered_map<unsigned int, Node*> id_to_node;
    unsigned short max_level = 0;
    std::pair<std::pair<vec,vec>, std::vector<unsigned int>> last_oclussion = {{{0,0,0},{0,0,0}},{}};

  public:
    std::vector<std::vector<OctreeOBB>> objects_by_level;
    Node* root;
    Octree(const std::vector<math::OBB>& objs, double scale_factor = 1.0);
    ~Octree(){ delete root;};
    std::vector<Node*> getlevel_left_to_right() const {return level_left_to_right;};
    unsigned int getTotalLevels() const {return max_level;};
    bool getOrdered(unsigned int level) { return isOrdered[level]; };

    void insertNode(Node* node_begin, Node* node_end);
    unsigned int getTotalObjectsSize() const;
    void reorder();
    void assureObjectsInLevel(unsigned int level);
    unsigned int insert(const unsigned int level, OctreeOBB& obj, Node* who);
    void remove(const unsigned int level, const unsigned int obj_index);
    unsigned int computeOcclusions(const math::Frustum& camera,
                           const vec& camera_pos,
                           const vec& camera_dir,
                           std::vector<unsigned int>* visible_objs);
};

#endif
