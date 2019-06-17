#include "octree.h"
// #include "node.h"

// return true if a is less than b
bool checkOrderLess(Node* a, Node* b){
  if(a == b)
    return false;

  if(a == nullptr)
    std::cerr << "[Octree] - Inserting null node A" << '\n';

  if(b == nullptr)
    std::cerr << "[Octree] - Inserting null node B" << '\n';

  auto& a_path = a->getPath();
  auto& b_path = b->getPath();

  // std::cout << "--------------------\n"
  //           << " a_path = ";
  // for(auto& it: a->getPath())
  //   std::cout << it << "-";
  //
  // std::cout << std::endl << " b_path = ";
  // for(auto& it: b->getPath())
  //   std::cout << it << "-";
  // std::cout << std::endl;
  //
  // std::cout << " a < b == ";

  unsigned int min_size = std::min(a_path.size(), b_path.size());
  bool equals = true;
  for(unsigned int i = 0;  i < min_size && equals; ++i){
    if(a_path[i] < b_path[i]){
      // std::cout << "true" << std::endl;
      return true;
    }
    else if(a_path[i] > b_path[i]){
      // std::cout << "false" << std::endl;
      return false;
    }
  }
  // std::cout <<  ((a_path.size() == min_size) ? "false" : "true") << std::endl;
  // if both are equal to this point
  return a_path.size() == min_size ? false : true;
}

Octree::Octree(const std::vector<math::OBB>& objs, double scale_factor): objects_by_level(MAX_LEVLELS){
  math::AABB region({0,0,0},{0,0,0});
  for(unsigned int j = 0; j < objs.size(); ++j){
    region.Enclose(objs[j]);
  }

  // std::cout << "[Octree] - root created " << region << std::endl;
  region.Scale({0,0,0}, scale_factor);
  std::cout << "[Octree] - root created " << region << std::endl;
  root = new Node(this, region, {});
  insertNode(root, root);

  for(unsigned int j = 0; j < objs.size(); j++){
    root->insert(objs[j], j);
  }
  reorder();
}

void Octree::reorder(){
  Node* current_node;
  unsigned int index = 0;
  std::pair<unsigned int,math::OBB> tmp;
  for(unsigned int level = 0; level < level_left_to_right.size(); ++level){
    current_node = level_left_to_right[level];
    if(!isOrdered[level]){
      isOrdered[level] = true;
      sort(objects_by_level[level].begin(), objects_by_level[level].end());
      // Iterate each level
      index = 0;
      while(current_node != nullptr && index < objects_by_level[level].size()){
        auto object_indexes = current_node->getObjectsIndexes();
        // std::cout  << "{";
        //
        // for(auto& path : current_node->getPath())
        //   std::cout << path << ",";

        // std::cout << "} [" << current_node->getID() << "]\n" ;
        for(unsigned int i = 0; i < object_indexes.size(); ++i){
          if(object_indexes[i] != index){
            auto& other_node = id_to_node[objects_by_level[level][index].getNodeID()];
            current_node->replace(i, index);
            other_node->replaceValue(index, object_indexes[i]);
            id_to_node[objects_by_level[level][index].getNodeID()] = current_node;

            // std::swap(objects_by_level[level][index],
            //           objects_by_level[level][object_indexes[i]]);

            // tmp = objects_by_level[level][index];
            // objects_by_level[level][index] = objects_by_level[level][object_indexes[i]];
            // objects_by_level[level][object_indexes[i]] = tmp;

            // objects_by_level[level][index].getNodeID() = current_node->getID();
            // objects_by_level[level][object_indexes[i]].getNodeID() = other_node->getID();

            // std::cout << objects_by_level[level][index].getNodeID() << " ";

          }
          else{
            // std::cout << "|" << objects_by_level[level][index].getNodeID() << "| ";
          }

          ++index;
        }
        // std::cout << "\n ";
        current_node = current_node->getRightSibling();
      }
      // if(index != objects_by_level[level].size()){
      //   std::cout << "Level_"<< level <<" not completed\n";
      //   exit(-1);
      // }
    }
    // std::cout << "\n";
  }
}

void Octree::assureObjectsInLevel(unsigned int level){
  if(level == max_level){
    // objects_by_level.resize(level+1);
    ++max_level;
    level_left_to_right.push_back(nullptr);
    isOrdered.emplace_back(false);
  }
  else if (level > objects_by_level.size()){
    std::cerr << "[Octree] - Error cannot ask for 2 level_left_to_right deeper" << std::endl;
  }
}

unsigned int Octree::insert(const unsigned int level, OctreeOBB& obj, Node* who){
  assureObjectsInLevel(level);
  objects_by_level[level].push_back(obj);
  id_to_node.insert({obj.getNodeID(), who});
  return (objects_by_level[level].size() -1);
}

void Octree::insertNode(Node* node_begin, Node* node_end){
  if(node_begin == nullptr)
    std::cerr << "[Octree] - Inserting null node_begin" << '\n';

  if(node_end == nullptr)
    std::cerr << "[Octree] - Inserting null node_end" << '\n';

  assureObjectsInLevel(node_begin->getLevel());
  isOrdered[node_begin->getLevel()] = false;
  if(level_left_to_right[node_begin->getLevel()] != nullptr){
    if(checkOrderLess(node_begin,level_left_to_right[node_begin->getLevel()])){
      level_left_to_right[node_begin->getLevel()]->setLeftSibling(node_end);
      node_end->setRightSibling(level_left_to_right[node_begin->getLevel()]);
      node_begin->setLeftSibling(nullptr);
      level_left_to_right[node_begin->getLevel()] = node_begin;
    }
    else{
      bool inserted = false;
      auto it = level_left_to_right[node_begin->getLevel()]->getRightSibling();
      // Iterate each node_begin in array until node_begin is lower than it
      while(it->getRightSibling() != nullptr && !inserted){
        if(checkOrderLess(node_begin, it)){
          // std::cout << node_begin->getID() << " < " << it->getID() << "\n";
          it->getLeftSibling()->setRightSibling(node_begin);
          node_begin->setLeftSibling(it->getLeftSibling());
          it->setLeftSibling(node_end);
          node_end->setRightSibling(it);
          inserted = true;
        }
        it = it->getRightSibling();
      }
      // std::cout << "Inserted = " << (inserted ? "true" : "false") << "\n";
      // If node_begin is larger than every other node_begin
      if(!inserted && it->getRightSibling() == nullptr){
        node_begin->setLeftSibling(it);
        it->setRightSibling(node_begin);
        node_end->setRightSibling(nullptr);
        inserted = true;
      }
    }
  }
  else{
    level_left_to_right[node_begin->getLevel()] = node_begin;
  }
}


unsigned int Octree::getTotalObjectsSize() const {
  unsigned int sum = 0;
  for(auto& it : objects_by_level){
      sum+=it.size();
  }
  return sum;
}

void Octree::remove(const unsigned int level, const unsigned int obj_index){
  // std::cout << "[Octree] - deleting from level " << level << std::endl;
    objects_by_level[level].erase(objects_by_level[level].begin()+obj_index);

    Node* nodes = level_left_to_right[level];
    while(nodes != nullptr){
      // std::cout  << " {";
      // for(auto& path : nodes->getPath())
      //   std::cout << path << ",";
      // std::cout << "} ";
      nodes->decrementIndexes(obj_index);
      nodes = nodes->getRightSibling();
    }
  // std::cout << std::endl << "--------" << std::endl;
};


unsigned int Octree::computeOcclusions(const math::Frustum& camera,
                               const vec& camera_pos,
                               const vec& camera_dir,
                               std::vector<unsigned int>* visible_objs){

  using namespace std::chrono;
  auto t1 = high_resolution_clock::now();

  if(last_oclussion.first.first.x  == camera_pos.x &&
     last_oclussion.first.first.y  == camera_pos.y &&
     last_oclussion.first.first.z  == camera_pos.z/* &&
     last_oclussion.first.second.x  == camera_dir.x &&
     last_oclussion.first.second.y  == camera_dir.y &&
     last_oclussion.first.second.z  == camera_dir.z*/){

    *visible_objs = last_oclussion.second;

    auto t2 = std::chrono::high_resolution_clock::now();
    auto time_span = duration_cast<duration<double>>(t2 - t1);
    // std::cout << "[Octree] - Oclussions computation  took " << (time_span.count()*1000) << " ms Cache HIT\n";

    return last_oclussion.second.size();
  }

  std::deque<std::pair<OctreeOBB,unsigned int>> objs_levels;
  std::vector<unsigned int> visibility_array;
  unsigned int level = level_left_to_right.size() - 1;
  unsigned int octree_prune = 0;
  unsigned int id = -1;
  auto comp = [&](auto& a, auto& b){return a.first < b.first;};


  for(auto it = level_left_to_right.rbegin(); it != level_left_to_right.rend();++it){

    auto it2 = *it;
    while (it2 != nullptr) {
      // if(it2->region.Intersects(camera)){
        for(auto index : it2->getObjectsIndexes()){
          ++octree_prune;
          if(it2->getParent() != nullptr)
            id = it2->getParent()->getID();
          else{
            id = 1;
          }
          objs_levels.emplace_back(objects_by_level[level][index], id);
        }
      // }
      it2 = it2->getRightSibling();
    }

    std::sort(objs_levels.begin(), objs_levels.end(), comp);

    // std::cout << "_Sending objects = [";
    // for(auto& it : objs_levels){
    //   std::cout << it.first.getNodeID() << ",";
    // }
    // std::cout << "]" << std::endl;
    // std::cout << "Sending objects = [";
    // for(auto& it : objs_levels){
    //   std::cout << it.first.getObjectID() << ",";
    // }
    // std::cout << "]" << std::endl;

    if(objs_levels.size() > 0){
      CudaQuery query(objs_levels);
      query.run(camera_pos, &visibility_array);
    }
    // std::cout << "Visible objects = [";
    // for(auto& it : visibility_array){
    //   std::cout << (bool)it << ",";
    // }
    // std::cout << "] - " <<  visibility_array.size() << std::endl;


    if(level != 0){
      // Merge with upper level if not last level
      unsigned int erased_elements = 0;
      for(unsigned int i = 0; i < visibility_array.size(); ++i){
        if(visibility_array[i]){
          // std::cout << "[ " << objs_levels[i-erased_elements].getNodeID() << " -> " << parent_id[i] << "]\n";
          objs_levels[i-erased_elements].first.setNodeID(objs_levels[i-erased_elements].second);
        }
        else{
          objs_levels.erase(objs_levels.begin()+i-erased_elements);
          ++erased_elements;
        }
      }
    }

    --level;
  }

  unsigned int erased_elements = 0;
  for(unsigned int i = 0; i < visibility_array.size(); ++i){
    if(visibility_array[i]){
      // std::cout << "[ " << objs_levels[i-erased_elements].getNodeID() << " -> " << parent_id[i] << "]\n";
      objs_levels[i-erased_elements].first.setNodeID(0);
    }
    else{
      objs_levels.erase(objs_levels.begin()+i-erased_elements);
      ++erased_elements;
    }
  }
  std::sort(objs_levels.begin(), objs_levels.end(), comp);

  // std::cout << "________________________\n";
  // std::cout << "_Sending objects = [";
  // for(auto& it : objs_levels){
  //   std::cout << it.first.getNodeID() << ",";
  // }
  // std::cout << "]" << std::endl;

  CudaQuery query(objs_levels);
  query.run(camera_pos, &visibility_array);

  // std::cout << "Visible objects = [";
  // for(auto& it : visibility_array){
  //   std::cout << (bool)it << ",";
  // }
  // std::cout << "] - " <<  visibility_array.size() << std::endl;

  // Add last visible objects that went up the tree
  visible_objs->clear();
  visible_objs->reserve(visibility_array.size());
  for(unsigned int i = 0; i < visibility_array.size(); ++i){
    if(visibility_array[i]){
      // std::cout << objs_levels[i].getObjectID() << "_ ";
      visible_objs->push_back(objs_levels[i].first.getObjectID());
    }
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  auto time_span = duration_cast<duration<double>>(t2 - t1);
  std::cout << "[Octree] - Oclussions computation  took " << (time_span.count()*1000) << " ms\n";
  std::cout << "[Octree] - Objects after occlusion " << visible_objs->size()
            << "   |   Original size " << getTotalObjectsSize() << "\n";

  std::sort(visible_objs->begin(), visible_objs->end());

  last_oclussion.first.first = camera_pos;
  last_oclussion.first.second = camera_dir;
  last_oclussion.second = *visible_objs;

  return octree_prune;
};
