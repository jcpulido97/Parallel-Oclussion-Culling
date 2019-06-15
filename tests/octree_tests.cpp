#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include <fstream>
#include <iomanip>

#include <stdlib.h>

#include <deque>
#include <unordered_set>
#include <random>

#include "../libs/MathGeoLib.h"
#include "octree.h"

TEST_CASE("Octree generation & subdivision controlled case", "[Octree]" ) {
    using namespace std;
    vector<math::OBB> objs(5);
    objs[0].SetFrom(math::Sphere({0,0,0}, 1));
    objs[1].SetFrom(math::Sphere({3,5,3}, 2));
    objs[2].SetFrom(math::Sphere({0,1,0}, 1));
    objs[3].SetFrom(math::Sphere({3,7,3}, 2));
    objs[4].SetFrom(math::Sphere({2,0,0}, 1));

    Octree oc(objs);

    bool sorted = true;
    unordered_set<unsigned int> id_set;
    unsigned int objs_size=0;
    double acumulated_size = 0.0;
    for(unsigned int i = 0 ; i < oc.getTotalLevels(); ++i){
      sorted = true;
      acumulated_size += (oc.objects_by_level[i].size()/(float)objs.size())*100.0 ;

      sorted = is_sorted(oc.objects_by_level[i].begin(), oc.objects_by_level[i].end());

      REQUIRE(sorted);

      for(unsigned int j = 0; j < oc.objects_by_level[i].size(); ++j){
        ++objs_size;
        if(id_set.count(oc.objects_by_level[i][j].getNodeID()) == 0){
          id_set.insert(oc.objects_by_level[i][j].getNodeID());
        }
        else if(oc.objects_by_level[i][j-1].getNodeID() != oc.objects_by_level[i][j].getNodeID()){
          sorted = false;
        }
      }

      REQUIRE(sorted);
    }
    REQUIRE(objs_size == objs.size());
}


TEST_CASE("Octree generation & subdivision 1000 objects", "[Octree]" ) {
    using namespace std;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    const int size = 1000;

    vector<math::OBB> objs(size);
    auto pos = math::vec();
    for(int i = 0; i < size; ++i){
      pos = math::vec(dis(gen)*100.0, dis(gen)*100.0, dis(gen)*100.0);
      objs[i].SetFrom(math::Sphere(pos, dis(gen)*5.0));
    }

    Octree oc(objs);

    bool sorted = true;
    unordered_set<unsigned int> id_set;
    unsigned int objs_size=0;
    double acumulated_size = 0.0;
    for(unsigned int i = 0 ; i < oc.getTotalLevels(); ++i){
      sorted = true;
      acumulated_size += (oc.objects_by_level[i].size()/(float)objs.size())*100.0 ;

      sorted = is_sorted(oc.objects_by_level[i].begin(), oc.objects_by_level[i].end());

      REQUIRE(sorted);

      for(unsigned int j = 0; j < oc.objects_by_level[i].size(); ++j){
        ++objs_size;
        std::cout << oc.objects_by_level[i][j].getNodeID() << " ";
        if(id_set.count(oc.objects_by_level[i][j].getNodeID()) == 0){
          id_set.insert(oc.objects_by_level[i][j].getNodeID());
        }
        else if(oc.objects_by_level[i][j-1].getNodeID() != oc.objects_by_level[i][j].getNodeID()){
          sorted = false;
          std::cout << oc.objects_by_level[i][j-1].getNodeID() << "!=" << oc.objects_by_level[i][j].getNodeID() << "\n";
        }
      }
      REQUIRE(sorted);
    }
    REQUIRE(objs_size == objs.size());
}
