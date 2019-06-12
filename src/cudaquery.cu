#include "cudaquery.cuh"

#define EPSILON 0.001
#define MINUS_EPSILON -EPSILON
#define SIZE sizeof(std::pair<unsigned int, vec[8]>)
// #define CUDA_PRINT

// camera_pos = vec3
// ocluder    = vec3[8]
// ocludee    = vec3[8]
//vec3[8] corners format
/*
     --------2--------6
     depth /|       / |
         /  | {1} /   |
--------3--------7    |
        |{3}| {4}|{2} |
        |   0----|----4
 height |  / {0} |{5}/
        |/       | /
--------1--------5
        |  width |
   Orden de caras = {0}CentroDelante, {1}Arriba, {2}Derecha, {3}Izquierda, {4}CentroDetras, {5}Abajo
*/

__device__ __forceinline__
void checkFaces(const vec& camera_pos,  const vec& camera_dir,
                vec* ocluder, vec* ocludee, unsigned int* visible){

  float vec_length;
  float cull_dot;
  float nomin, denom, prod3;
  float face_area;
  float parcial_area_intersection;
  vec face;
  vec cull;
  vec intersection;
  vec diff, scaled;
  vec ray_vector;
  vec ab, ac;
  vec ABAC_cross;
  vec a, b, c, d;

  switch (threadIdx.x) {
    case 0:
      a = ocluder[3];
      b = ocluder[7];
      c = ocluder[5];
      d = ocluder[1];
    break;
    case 1:
      a = ocluder[3];
      b = ocluder[2];
      c = ocluder[6];
      d = ocluder[7];
    break;
    case 2:
      a = ocluder[7];
      b = ocluder[6];
      c = ocluder[4];
      d = ocluder[5];
    break;
    case 3:
      a = ocluder[2];
      b = ocluder[3];
      c = ocluder[1];
      d = ocluder[0];
    break;
    case 4:
      a = ocluder[6];
      b = ocluder[2];
      c = ocluder[0];
      d = ocluder[4];
    break;
    case 5:
      a = ocluder[4];
      b = ocluder[0];
      c = ocluder[1];
      d = ocluder[5];
    break;
  }

  ab = {b.x - a.x,
        b.y - a.y,
        b.z - a.z};

  ac = {c.x - a.x,
        c.y - a.y,
        c.z - a.z};

  ///////// normal computation ///////
  face = {ab.y * ac.z - ab.z * ac.y,
          ab.z * ac.x - ab.x * ac.z,
          ab.x * ac.y - ab.y * ac.x};
  ///////////////////////////////////

  /////// vector normalization //////
  vec_length = sqrt((face.x*face.x) +
                    (face.y*face.y) +
                    (face.z*face.z));

  face.x = face.x /vec_length;
  face.y = face.y /vec_length;
  face.z = face.z /vec_length;

  face_area = vec_length;
  ///////////////////////////////////

  // Check back face culling as OpenGL
  // https://en.wikipedia.org/wiki/Back-face_culling
  cull.x = a.x - camera_pos.x;
  cull.y = a.y - camera_pos.y;
  cull.z = a.z - camera_pos.z;

  cull_dot = (cull.x*face.x) +
             (cull.y*face.y) +
             (cull.z*face.z);

  if(cull_dot >= EPSILON){
    for(unsigned int i = 0; i < 8; ++i){
      if(visible[i]){
        // Intersection computation
        // https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#C.2B.2B
        ray_vector = {camera_pos.x - ocludee[i].x,
                      camera_pos.y - ocludee[i].y,
                      camera_pos.z - ocludee[i].z};

        diff = {a.x - ocludee[i].x,
                a.y - ocludee[i].y,
                a.z - ocludee[i].z};

        nomin = diff.x * face.x +
                diff.y * face.y +
                diff.z * face.z;

        denom = ray_vector.x * face.x +
                ray_vector.y * face.y +
                ray_vector.z * face.z;

        // prod3 should not be 0.0 or close and always positive so it is behind the plane
        if(!(denom >= MINUS_EPSILON &&  EPSILON >= denom)){
          prod3 = nomin / denom;
          if(prod3 >= MINUS_EPSILON){

            scaled = {ray_vector.x * prod3,
                      ray_vector.y * prod3,
                      ray_vector.z * prod3};

            intersection.x = ocludee[i].x + scaled.x;
            intersection.y = ocludee[i].y + scaled.y;
            intersection.z = ocludee[i].z + scaled.z;

            #ifdef CUDA_PRINT
            printf("%i %i Plane gpu_points [%f,%f,%f] [%f,%f,%f] [%f,%f,%f]\n", blockIdx.x , threadIdx.x,
                                        a.x,a.y,a.z,
                                        b.x,b.y,b.z,
                                        c.x,c.y,c.z);

            printf("%i %i Plane normal [%f,%f,%f]\n", blockIdx.x , threadIdx.x,
                                            face.x,face.y,face.z);

            printf("%i %i Ray vector [%f,%f,%f] - Ray point [%f,%f,%f]\n", blockIdx.x , threadIdx.x,
                    ray_vector.x,ray_vector.y,ray_vector.z, ocludee[i].x,ocludee[i].y,ocludee[i].z);

            printf("%i %i Intersection [%f,%f,%f]\n", blockIdx.x , threadIdx.x,
              intersection.x,intersection.y,intersection.z);
            #endif

            parcial_area_intersection = 0.0;
            // Point must be inside the box if area of triangles to corners is
            // equal to the total face area
            // a = intersection_point
            // B and C = every other corner
            // Extension of 2D problem in web
            // https://www.geeksforgeeks.org/check-whether-given-point-lies-inside-rectangle-not/
            // https://www.quora.com/How-can-I-find-the-area-of-a-triangle-in-3D-coordinate-geometry
            ab = {a.x - intersection.x,
                  a.y - intersection.y,
                  a.z - intersection.z};

            ac = {b.x - intersection.x,
                  b.y - intersection.y,
                  b.z - intersection.z};

            ABAC_cross = {ab.y * ac.z - ab.z * ac.y,
                          ab.z * ac.x - ab.x * ac.z,
                          ab.x * ac.y - ab.y * ac.x};

            vec_length = sqrt((ABAC_cross.x*ABAC_cross.x) +
                              (ABAC_cross.y*ABAC_cross.y) +
                              (ABAC_cross.z*ABAC_cross.z));

            parcial_area_intersection += vec_length;
            // #ifdef CUDA_PRINT
            // printf("%i %i %i Parcial += %f -> %f \n", blockIdx.x, threadIdx.x, i, vec_length, parcial_area_intersection);
            // #endif

            ab = {b.x - intersection.x,
                  b.y - intersection.y,
                  b.z - intersection.z};

            ac = {c.x - intersection.x,
                  c.y - intersection.y,
                  c.z - intersection.z};

            ABAC_cross = {ab.y * ac.z - ab.z * ac.y,
                          ab.z * ac.x - ab.x * ac.z,
                          ab.x * ac.y - ab.y * ac.x};

            vec_length = sqrt((ABAC_cross.x*ABAC_cross.x) +
                              (ABAC_cross.y*ABAC_cross.y) +
                              (ABAC_cross.z*ABAC_cross.z));

            parcial_area_intersection += vec_length;
            // #ifdef CUDA_PRINT
            // printf("%i %i %i Parcial += %f -> %f \n", blockIdx.x, threadIdx.x, i, vec_length, parcial_area_intersection);
            // #endif

            ab = {c.x - intersection.x,
                  c.y - intersection.y,
                  c.z - intersection.z};

            ac = {d.x - intersection.x,
                  d.y - intersection.y,
                  d.z - intersection.z};

            ABAC_cross = {ab.y * ac.z - ab.z * ac.y,
                          ab.z * ac.x - ab.x * ac.z,
                          ab.x * ac.y - ab.y * ac.x};

            vec_length = sqrt((ABAC_cross.x*ABAC_cross.x) +
                              (ABAC_cross.y*ABAC_cross.y) +
                              (ABAC_cross.z*ABAC_cross.z));

            parcial_area_intersection += vec_length;
            // #ifdef CUDA_PRINT
            // printf("%i %i %i Parcial += %f -> %f \n", blockIdx.x, threadIdx.x, i, vec_length, parcial_area_intersection);
            // #endif

            ab = {d.x - intersection.x,
                  d.y - intersection.y,
                  d.z - intersection.z};

            ac = {a.x - intersection.x,
                  a.y - intersection.y,
                  a.z - intersection.z};


            ABAC_cross = {ab.y * ac.z - ab.z * ac.y,
                          ab.z * ac.x - ab.x * ac.z,
                          ab.x * ac.y - ab.y * ac.x};

            vec_length = sqrt((ABAC_cross.x*ABAC_cross.x) +
                              (ABAC_cross.y*ABAC_cross.y) +
                              (ABAC_cross.z*ABAC_cross.z));

            parcial_area_intersection += vec_length;
            // #ifdef CUDA_PRINT
            // printf("%i %i %i Parcial += %f -> %f \n", blockIdx.x, threadIdx.x, i, vec_length, parcial_area_intersection);
            // #endif

            // Multiply ommited 0.5 in every triangle
            parcial_area_intersection *= 0.5;
            // Multiply face_area by 2 is the same as divide by 2 parcial_area_intersection
            // face_area *= 2;

            float area_difference = parcial_area_intersection - face_area;

            #ifdef CUDA_PRINT
            printf("%i %i %i Parcial -> %f - %f <- Total == %f\n", blockIdx.x, threadIdx.x, i, parcial_area_intersection, face_area, area_difference);
            #endif

            if(MINUS_EPSILON <= area_difference  && area_difference <= EPSILON){
              #ifdef CUDA_PRINT
              printf("%i %i Point OCLUDED \n", blockIdx.x, threadIdx.x);
              #endif
              // Point lies inside rectangle so it's ocluded
              atomicAnd(&visible[i], false);
            }
          }
          // else{
          //   #ifdef CUDA_PRINT
          //   printf("%i %i Point not behind\n", blockIdx.x, threadIdx.x);
          //   #endif
          // }
        }
      }
    }
  }
  // else{
  //   #ifdef CUDA_PRINT
  //   printf("%i %i Not visible\n", blockIdx.x, threadIdx.x);
  //   #endif
  // }
}

__global__
void occlusion(vec camera_pos, vec camera_dir, std::pair<unsigned int,vec[8]>* gpu_points, unsigned int* object_visibility_array, unsigned int size) {
  int object_to_check = blockIdx.x;
  int i = object_to_check + 1;
  __shared__ unsigned int visibility_points_array[8];

  if(threadIdx.x == 0){
    visibility_points_array[0] = 1;
    visibility_points_array[1] = 1;
    visibility_points_array[2] = 1;
    visibility_points_array[3] = 1;
    visibility_points_array[4] = 1;
    visibility_points_array[5] = 1;
    visibility_points_array[6] = 1;
    visibility_points_array[7] = 1;
  }

  __syncthreads();

  if(i < size){
    // Check if gpu_points are from same node as our
    while(gpu_points[i].first == gpu_points[object_to_check].first){

      checkFaces(camera_pos, camera_dir, gpu_points[object_to_check].second, gpu_points[i].second, visibility_points_array);

      // Every thread must have completed its face checking
      __syncthreads();

      if(threadIdx.x == 0){
        #ifdef CUDA_PRINT
          printf("---------------Object %i against %i\n", object_to_check, i);
        #endif
        unsigned int any_visible = 0;
        #pragma unroll
        for (unsigned int p = 0; p < 8 && !any_visible; ++p) {
          any_visible |= visibility_points_array[p];
        }
        // If every point is not visible then object is ocluded
        if(!any_visible){
          #ifdef CUDA_PRINT
          printf("---------------Object %i OCLUDED by %i\n", i, object_to_check);
          #endif
          atomicAnd(&object_visibility_array[i],0);
        }
        else{
          // #ifdef CUDA_PRINT
          // printf("---------------Object %i DON'T oclude %i\n", object_to_check, i);
          // #endif
        }
        // Restart visibility_points_array for next object
        visibility_points_array[0] = 1;
        visibility_points_array[1] = 1;
        visibility_points_array[2] = 1;
        visibility_points_array[3] = 1;
        visibility_points_array[4] = 1;
        visibility_points_array[5] = 1;
        visibility_points_array[6] = 1;
        visibility_points_array[7] = 1;
      }

      // All threads finish and start the loop together
      __syncthreads();

      if((++i) == size)
        break;
    }
  }

  i = object_to_check - 1;

  if(i >= 0){
    // Check if gpu_points are from same node as our
    while(gpu_points[i].first == gpu_points[object_to_check].first){

      checkFaces(camera_pos, camera_dir, gpu_points[object_to_check].second, gpu_points[i].second, visibility_points_array);

      __syncthreads();

      if(threadIdx.x == 0){
        #ifdef CUDA_PRINT
          printf("---------------Object %i against %i\n", object_to_check, i);
        #endif
        unsigned int any_visible = false;
        #pragma unroll
        for (unsigned int p = 0; p < 8 && !any_visible; ++p) {
          any_visible |= visibility_points_array[p];
        }
        // If every point is not visible then object is ocluded
        if(!any_visible){
          #ifdef CUDA_PRINT
          printf("---------------Object %i OCLUDED by %i\n", i, object_to_check);
          #endif
          atomicAnd(&object_visibility_array[i],0);
        }
        else{
          // #ifdef CUDA_PRINT
          // printf("---------------Object %i DON'T oclude %i\n", object_to_check, i);
          // #endif
        }
        // Restart visibility_points_array for next object
        visibility_points_array[0] = 1;
        visibility_points_array[1] = 1;
        visibility_points_array[2] = 1;
        visibility_points_array[3] = 1;
        visibility_points_array[4] = 1;
        visibility_points_array[5] = 1;
        visibility_points_array[6] = 1;
        visibility_points_array[7] = 1;
      }

      // All threads finish and start the loop together
      __syncthreads();

      if((--i) < 0)
        break;
    }
  }
}

CudaQuery::CudaQuery(const std::deque<OctreeOBB>& objects){
  size = objects.size();
  cpu_points = new std::pair<unsigned int, vec[8]>[size];
  math::vec tmp_points[8];
  for(int i = 0; i < objects.size(); ++i){
    cpu_points[i].first = objects[i].getNodeID();
    objects[i].obb.GetCornerPoints(tmp_points);

    // std::cout << "{" << tmp_points[0].x << ',' << tmp_points[0].y << ',' << tmp_points[0].z << "}\n";
    // std::cout << "{" << tmp_points[1].x << ',' << tmp_points[1].y << ',' << tmp_points[1].z << "}\n";
    // std::cout << "{" << tmp_points[2].x << ',' << tmp_points[2].y << ',' << tmp_points[2].z << "}\n";
    // std::cout << "{" << tmp_points[3].x << ',' << tmp_points[3].y << ',' << tmp_points[3].z << "}\n";
    // std::cout << "{" << tmp_points[4].x << ',' << tmp_points[4].y << ',' << tmp_points[4].z << "}\n";
    // std::cout << "{" << tmp_points[5].x << ',' << tmp_points[5].y << ',' << tmp_points[5].z << "}\n";
    // std::cout << "{" << tmp_points[6].x << ',' << tmp_points[6].y << ',' << tmp_points[6].z << "}\n";
    // std::cout << "{" << tmp_points[7].x << ',' << tmp_points[7].y << ',' << tmp_points[7].z << "}\n------------\n";

    cpu_points[i].second[0] = {tmp_points[0].x, tmp_points[0].y, tmp_points[0].z};
    cpu_points[i].second[1] = {tmp_points[1].x, tmp_points[1].y, tmp_points[1].z};
    cpu_points[i].second[2] = {tmp_points[2].x, tmp_points[2].y, tmp_points[2].z};
    cpu_points[i].second[3] = {tmp_points[3].x, tmp_points[3].y, tmp_points[3].z};
    cpu_points[i].second[4] = {tmp_points[4].x, tmp_points[4].y, tmp_points[4].z};
    cpu_points[i].second[5] = {tmp_points[5].x, tmp_points[5].y, tmp_points[5].z};
    cpu_points[i].second[6] = {tmp_points[6].x, tmp_points[6].y, tmp_points[6].z};
    cpu_points[i].second[7] = {tmp_points[7].x, tmp_points[7].y, tmp_points[7].z};
  }
  if(gpu_points == nullptr){
    gpuErrchk(cudaFree(gpu_points));
    gpu_points = nullptr;
    object_visibility_array = nullptr;
  }
  unsigned int one = 4294967295;
  gpuErrchk(cudaMalloc(&object_visibility_array, size*sizeof(unsigned int)));
  gpuErrchk(cudaMemset(object_visibility_array, one, size*sizeof(unsigned int)));
  gpuErrchk(cudaMalloc(&gpu_points, size*SIZE));
  gpuErrchk(cudaMemcpy(gpu_points, cpu_points, size*SIZE, cudaMemcpyHostToDevice));
}


void CudaQuery::transferGPU(){
  if(gpu_points != nullptr){
    gpuErrchk(cudaMalloc(&gpu_points, size*SIZE));
  }
  gpuErrchk(cudaMemcpy(gpu_points, cpu_points, size*SIZE, cudaMemcpyHostToDevice));
}

void CudaQuery::run(vec camera_pos, vec camera_dir, std::vector<unsigned int>* cpu_visibility){
  if(size > 0){
    if(gpu_points != nullptr){
      using namespace std::chrono;
      auto t1 = high_resolution_clock::now();

      occlusion<<<size,6>>>(camera_pos, camera_dir, gpu_points, object_visibility_array, size);
      gpuErrchk(cudaDeviceSynchronize());

      // Finished
      cpu_visibility->resize(size);
      gpuErrchk(cudaMemcpy(cpu_visibility->data(), object_visibility_array, size*sizeof(unsigned int), cudaMemcpyDeviceToHost));

      auto t2 = std::chrono::high_resolution_clock::now();
      auto time_span = duration_cast<duration<double>>(t2 - t1);

      std::cout << "[CudaQuery] - Computation of " << size <<" objects took " << (time_span.count()*1000) << " ms\n";
    }
    else{
      std::cerr << "[CudaQuery] - Trying to run query before allocating" << std::endl;
    }
  }
};
