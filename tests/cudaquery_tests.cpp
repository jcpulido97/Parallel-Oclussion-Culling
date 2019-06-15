#include "catch.hpp"

#include <deque>
#include <vector>

#include "cudaquery.cuh"
#include "octree.h"

TEST_CASE("Occlusions culling computation", "[CudaQuery]" ) {
    using namespace std;
    vector<math::OBB> objs(3);
    objs[0].SetFrom(math::Sphere({0,0,0}, 2));         // ocluder
    objs[1].SetFrom(math::Sphere({0,0,0}, 1));         // ocluder
    objs[2].SetFrom(math::Sphere({6,0,0}, 1));         // Separated object

    deque<std::pair<OctreeOBB,unsigned int>> obb;
    for(auto& it : objs)
      obb.emplace_back(OctreeOBB(0, 0, it, {0}), 1);

    CudaQuery cu(obb);

    vec camera_pos = {0,0,5};
    vec camera_dir;
    math::vec camera_dir_normal = {0,0,-1};
    camera_dir = {camera_dir_normal.x,camera_dir_normal.y,camera_dir_normal.z};
    math::vec camera_up = {0,1,0};
    math::Frustum frustum;
    frustum.SetPos({camera_pos.x,camera_pos.y,camera_pos.z});
    frustum.SetFront({camera_dir.x,camera_dir.y,camera_dir.z});
    frustum.SetUp(camera_up);
    frustum.SetKind(math::FrustumSpaceGL,math::FrustumRightHanded);
    frustum.SetViewPlaneDistances(1.0, 10000.0);
    frustum.SetPerspective(10.0, 10.0);

    vector<unsigned int> vis;
    cu.run(camera_pos, &vis);
    vector<unsigned int> visible_objs;
    unsigned int i = 0;
    for(auto& it : vis){
      if(it){
        visible_objs.push_back(i++);
      }
    }
    REQUIRE(visible_objs.size() == 2);
}


TEST_CASE("Occlusions Octree culling computation", "[CudaQuery]" ) {
    using namespace std;
    vector<math::OBB> objs(3);
    objs[0].SetFrom(math::Sphere({0,0,0}, 2));         // ocluder
    objs[1].SetFrom(math::Sphere({0,0,0}, 1));         // ocluder
    objs[2].SetFrom(math::Sphere({6,0,0}, 1));         // Separated object

    Octree cu(objs);

    vec camera_pos = {0,0,5};
    vec camera_dir;
    math::vec camera_dir_normal = {0,0,-1};
    camera_dir = {camera_dir_normal.x,camera_dir_normal.y,camera_dir_normal.z};
    math::vec camera_up = {0,1,0};
    math::Frustum frustum;
    frustum.SetPos({camera_pos.x,camera_pos.y,camera_pos.z});
    frustum.SetFront({camera_dir.x,camera_dir.y,camera_dir.z});
    frustum.SetUp(camera_up);
    frustum.SetKind(math::FrustumSpaceGL,math::FrustumRightHanded);
    frustum.SetViewPlaneDistances(1.0, 10000.0);
    frustum.SetPerspective(10.0, 10.0);

    vector<unsigned int> vis;
    cu.computeOcclusions(frustum, camera_pos, camera_dir, &vis);
    REQUIRE(vis.size() == 2);
}


TEST_CASE("Occlusions Octree culling computation Big numbers", "[CudaQuery]" ) {
    using namespace std;
    vector<math::OBB> objs(3);
    objs[0].SetFrom(math::Sphere({0,0,-1000}, 100));         // ocluder
    objs[1].SetFrom(math::Sphere({0,0,-120000}, 99));         // ocluder
    objs[2].SetFrom(math::Sphere({6,0,0}, 1));         // Separated object

    Octree cu(objs);

    vec camera_pos = {0,0,5};
    vec camera_dir;
    math::vec camera_dir_normal = {0,0,-1};
    camera_dir = {camera_dir_normal.x,camera_dir_normal.y,camera_dir_normal.z};
    math::vec camera_up = {0,1,0};
    math::Frustum frustum;
    frustum.SetPos({camera_pos.x,camera_pos.y,camera_pos.z});
    frustum.SetFront({camera_dir.x,camera_dir.y,camera_dir.z});
    frustum.SetUp(camera_up);
    frustum.SetKind(math::FrustumSpaceGL,math::FrustumRightHanded);
    frustum.SetViewPlaneDistances(1.0, 10000.0);
    frustum.SetPerspective(10.0, 10.0);

    vector<unsigned int> vis;
    cu.computeOcclusions(frustum, camera_pos, camera_dir, &vis);
    REQUIRE(vis.size() == 2);
    REQUIRE(vis[0] == 0);
    REQUIRE(vis[1] == 2);
}
