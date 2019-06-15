#pragma once

#include <vector>
#include "../libs/MathGeoLib.h"

class OctreeOBB{
  private:
    unsigned int objID;
    unsigned int nodeID;
    std::vector<unsigned int> nodePath;
  public:
    math::OBB obb;
    OctreeOBB(unsigned int vector_index, unsigned int id, math::OBB ob, std::vector<unsigned int> path):
        objID{vector_index}, nodeID{id}, nodePath{path}, obb{ob}{};
    unsigned int getNodeID() const {return nodeID; };
    void setNodeID(unsigned int id){ nodeID = id;};
    unsigned int getObjectID() const  {return objID;};
    void setPath(std::vector<unsigned int> _nodePath) { nodePath = _nodePath;};

    friend bool operator<(const OctreeOBB& a, const OctreeOBB& b){
      unsigned int min_size = std::min(a.nodePath.size(), b.nodePath.size());
      bool equals = true;
      for(unsigned int i = 0;  i < min_size && equals; ++i){
        if(a.nodePath[i] < b.nodePath[i]){
          return true;
        }
        else if(a.nodePath[i] > b.nodePath[i]){
          return false;
        }
      }
      // if both are equal to this point
      return a.nodePath.size() == min_size ? false : true;
    }

};
