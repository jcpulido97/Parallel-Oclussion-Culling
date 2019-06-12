#include "node.h"
#include "octree.h"

unsigned int Node::class_id = 0;

void Node::insert(const unsigned int index){
  objects.emplace_back(index);
};

void Node::insert(math::OBB obj){
  if(divided){
    bool inserted = false;
    for(auto& it :children){
      if(it.region.Contains(obj)){
        it.insert(obj);
        inserted = true;
        break;
      }
    }
    if(!inserted){
        int obj_index;
        OctreeOBB pair = {getID(), math::OBB(obj), getPath()};
        obj_index = octree->insert(level, pair, this);
        insert(obj_index);
    }
  }
  else{
    if(region.Contains(obj)){
      int obj_index;
      OctreeOBB pair = {getID(), math::OBB(obj), getPath()};
      obj_index = octree->insert(level, pair, this);
      insert(obj_index);
    }
    if(objects.size() > MAX_OBJECTS){
      // std::cout << "[Node] - Subdivide: "<< getID() <<"\n";
      subdivide();
    }
  }
};

bool Node::remove(const math::OBB& obj){
  auto& vec = octree->objects_by_level[level];
  // std::cout << "[Node] - size: "<< vec.size() <<"\n";
  for(unsigned int i = 0; i < vec.size(); ++i){
    if(&(vec[i].obb) == &obj){
      return remove(i);
    }
  }
  return false;
};

void Node::decrementIndexes(const unsigned int threadshold){
  for(unsigned int i = 0; i < objects.size(); ++i){
    if(objects[i] > threadshold){
      // Update indexes to new objects positions
      --objects[i];
    }
  }
}

bool Node::remove(const unsigned int obj_index){
  bool borrado = false;

  // Iterate our objects
  for(unsigned int i = 0; i < objects.size() && !borrado; ++i){
    if(objects[i] == obj_index){
      objects.erase(objects.begin()+i);
      octree->remove(this->level, obj_index);
      borrado = true;
    }
  }

  return borrado;
};

void Node::subdivide(){
  if(!divided){
    divided = true;
    int lvl = level + 1;
    octree->assureObjectsInLevel(lvl);

    math::vec center = region.CenterPoint();

    math::AABB octant[8];
    octant[0].minPoint = region.minPoint;
    octant[0].maxPoint = center;
    octant[1].minPoint = math::vec(center.x, region.minPoint.y, region.minPoint.z);
    octant[1].maxPoint = math::vec(region.maxPoint.x, center.y, center.z);
    octant[2].minPoint = math::vec(center.x, region.minPoint.y, center.z);
    octant[2].maxPoint = math::vec(region.maxPoint.x, center.y, region.maxPoint.z);
    octant[3].minPoint = math::vec(region.minPoint.x, region.minPoint.y, center.z);
    octant[3].maxPoint = math::vec(center.x, center.y, region.maxPoint.z);
    octant[4].minPoint = math::vec(region.minPoint.x, center.y, region.minPoint.z);
    octant[4].maxPoint = math::vec(center.x, region.maxPoint.y, center.z);
    octant[5].minPoint = math::vec(center.x, center.y, region.minPoint.z);
    octant[5].maxPoint = math::vec(region.maxPoint.x, center.y, center.z);
    octant[6].minPoint = center;
    octant[6].maxPoint = region.maxPoint;
    octant[7].minPoint = math::vec(region.minPoint.x, center.y, center.z);
    octant[7].maxPoint = math::vec(center.x, region.maxPoint.y, region.maxPoint.z);

    // std::cout << "[Node] - Octants_"<< level <<" = \n";
    // for(auto& it: octant)
    //   std::cout << it << std::endl;

    auto copy_id =path;
    // std::cout << "[Node] - Subdivision Level_"<< level <<" = {";
    // for(auto& it: copy_id)
    //   std::cout << it << "-";
    // std::cout << "0/7}" << std::endl;
    copy_id.push_back(0);
    children.push_back(Node(octree, octant[0], copy_id, lvl, this));
    copy_id[copy_id.size()-1] = 1;
    children.push_back(Node(octree, octant[1], copy_id, lvl, this));
    copy_id[copy_id.size()-1] = 2;
    children.push_back(Node(octree, octant[2], copy_id, lvl, this));
    copy_id[copy_id.size()-1] = 3;
    children.push_back(Node(octree, octant[3], copy_id, lvl, this));
    copy_id[copy_id.size()-1] = 4;
    children.push_back(Node(octree, octant[4], copy_id, lvl, this));
    copy_id[copy_id.size()-1] = 5;
    children.push_back(Node(octree, octant[5], copy_id, lvl, this));
    copy_id[copy_id.size()-1] = 6;
    children.push_back(Node(octree, octant[6], copy_id, lvl, this));
    copy_id[copy_id.size()-1] = 7;
    children.push_back(Node(octree, octant[7], copy_id, lvl, this));

    for(int i = 1; i < 7; ++i){
      children[i].left_sibling  = &children[i-1];
      children[i].right_sibling = &children[i+1];
    }
    children[0].right_sibling = &children[1];
    children[0].left_sibling = nullptr;
    children[7].left_sibling  = &children[6];
    children[7].right_sibling = nullptr;

    octree->insertNode(&children[0], &children[7]);

    auto& vec = octree->objects_by_level[level];
    bool inserted = false;
    unsigned int true_size = objects.size();
    // Iterate our objects
    for(unsigned int obj = 0; obj < true_size; ++obj){
      // Iterate octants to see where to put each object
      inserted = false;
      for(unsigned int i = 0; i < 8 && !inserted; ++i){
        if(octant[i].Contains(vec[objects[obj]].obb)){
          // std::cout << "[Node] - Item moved " << octree->assureObjectsInLevel(level)[it].first << std::endl;
          children[i].insert(vec[objects[obj]].obb);
          remove(objects[obj--]);
          true_size--;
          inserted = true;
        }
      }
    }
  }
};

unsigned int Node::nodeCount(){
  if(divided){
    unsigned int count = 1;
    for(auto& it :children)
      count+=it.nodeCount();
    return count;
  }
  else{
    return 1;
  };
};

bool Node::replace(const unsigned int index, const unsigned int val){
  if(objects.size() < index){
    std::cerr << "[Node] - Replacing item OutOfArray : " << index << " " << val << std::endl;
    return false;
  }
  else{
    objects[index] = val;
    return true;
  }
};

bool Node::replaceValue(const unsigned int old_val, const unsigned int new_val){
  for(unsigned int i = 0; i < objects.size(); ++i){
    if(objects[i] == old_val){
      objects[i] = new_val;
      return true;
    }
  }
  return false;
};
