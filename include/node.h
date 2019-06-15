#ifndef NODE_H_
#define NODE_H_

#include <vector>
#include <stdint.h>
#include <string>

//#include "LibFwd.h"
//#include "octree.h"

#include "octreeobb.h"
#include "../libs/MathGeoLib.h"

class Octree;

class Node {
  // friend class Octree;
  public:
    // static const uint8_t MIN_SIZE    = 1;      // Minimun Size of Region 1x1x1
    static const uint8_t MAX_OBJECTS = 4;      // Maximum amount of objects per cell
    const math::AABB region;

  protected:
    static unsigned int class_id;
    const unsigned int id;
    Octree* octree;
    std::vector<unsigned int> path;
    const unsigned short level;
    bool divided;
    std::vector<Node> children;
    std::vector<unsigned int> objects;
    Node* parent;
    Node* left_sibling;
    Node* right_sibling;

    void subdivide();

  public:
    unsigned int nodeCount();
    Node(Octree* octree,
         math::AABB box,
         std::vector<unsigned int> path,
         uint8_t lvl = 0,
         Node* father = nullptr,
         Node* l_sibling = nullptr,
         Node* r_sibling = nullptr)
           :region{box}, id{++class_id}, octree{octree}, path(path), level{lvl},
            divided{false}, parent{father},
            left_sibling{l_sibling}, right_sibling{r_sibling}
            {
              // if(octree == nullptr){
              //   std::cerr << "octree = nullptr " << std::endl;
              //   exit(-1);
              // }
              // std::cout << "{";
              // std::cout << "parent = " << father << std::endl;
              // std::cout << "l_sibling = " << l_sibling << std::endl;
              // std::cout << "r_sibling = " << r_sibling << std::endl;
              // for(auto& it:path)
              //   std::cout << it << "-";
              // std::cout << "}" << std::endl;
            };

    bool isDivided()                              const {return divided;};
    math::AABB getRegion()                        const {return region;};
    Node* getParent()                             const {return parent;};
    Node* getLeftSibling()                        const {return left_sibling;};
    Node* getRightSibling()                       const {return right_sibling;};
    unsigned short getLevel()                     const {return level;};
    const std::vector<unsigned int>& getPath()    const {return path;};
    unsigned int getID()                          const {return id;};
    std::vector<unsigned int> getObjectsIndexes() const {return objects;};
    // std::vector<std::vector<std::pair<unsigned int,math::OBB>>>*  getObjectsArray() const {return objects_array;}
    void setLeftSibling(Node* other)  {left_sibling = other;};
    void setRightSibling(Node* other) {right_sibling = other;};
    std::vector<Node>& getChildren() {return children;};

    void insert(OctreeOBB obj);
    void insert(math::OBB obj, unsigned int vector_index);
    void insert(const unsigned int index);
    bool remove(const math::OBB& obj);
    bool remove(const unsigned int index);
    bool replace(const unsigned int index, const unsigned int val);
    bool replaceValue(const unsigned int old_val, const unsigned int new_val);
    void decrementIndexes(const unsigned int threshold);
};

#endif
