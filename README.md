# Framework for abstraction of heterogeneous CPU / GPU environments

**Using the MathGeoLib library https://github.com/juj/MathGeoLib**

### Dependencies

```bash
 $> sudo apt-get install nvidia-cuda-toolkit libglew-dev freeglut3 freeglut3-dev libglfw3 libglfw3-dev
 $> make
```

## Video of demo
[![Occlusion-Culling](https://i.imgur.com/uopEtAv.jpeg)](https://vimeo.com/348480920 "Occlusion-Culling")

## Main Data Structure (Octree optimized)
![tree](https://raw.githubusercontent.com/jcpulido97/Parallel-Oclussion-Culling/master/doc/img/arbol.png)

**Documentation of data structures (spanish)** https://github.com/jcpulido97/Parallel-Oclussion-Culling/blob/master/doc/presentacion.pdf

## Rendering optimization by parallel culling occlusion calculation

In this project, a framework will be developed that provides computing resources to applications in a way that allows them to isolate themselves from low-level details while providing them with tools to adapt the allocation of resources to their real needs. In this case we will focus on trying to optimize the optimization of the scene rendering process, ignoring those objects that are hidden by others so that time is not wasted on calculations that will not be really useful later.

This work comes from the motivation caused by the lack of open content that exists in the field of graphics, although it is true that there are libraries that help to carry out this task, all are usually closed source or solutions that have been implemented directly from the developers of the different graphics engines that are currently used, either Unreal Engine or Unity. Which, even though they are free to use, we cannot know what algorithms / technologies compose them or if they can be improved or adapted for the next technologies.

![main](https://raw.githubusercontent.com/jcpulido97/TFG/master/doc/img/screenshot.png?token=AFM4SFGQLZNL2OPI63JE7AC5CCQRA)

### Structure

- **doc/** you will find all the project documentation, both diagrams and images
- **include/** all project declaration files
- **src/** all project implementation files
- **tests/** all the tests of the different classes of the project
- **libs/** the libraries used in this project. In this case only [MathGeoLib](https://github.com/juj/MathGeoLib)


### Controls

| Use | Key |
| ----------------------------- | ---------------------- |
| **Move Camera** | W A S D |
| **Rotate Camera** | ↑ ↓ ← → (Arrow Keys) |
| **Up / Down** | PageUp / PageDown |
| **Occlussion Culling Toggle** | or |

### Improvements
Different tests have been run on the following equipment:

| Category | Device |
| --------------------- | ---------------------- |
| **CPU** | i7-7700HQ |
| **Amount of RAM** | 16 GB |
| **GPU** | NVIDIA GTX 1050 |
| **Operating System** | Ubuntu 18.04 |
| **GPU Drivers** | NVIDIA drivers 390.116 |

![Improvements](https://raw.github.com/jcpulido97/TFG/master/doc/img/prune_benchmark.svg?sanitize=true)

You get quite a few improvements using the software to avoid rendering objects that are occluded

![teoric_limit](https://raw.github.com/jcpulido97/TFG/master/doc/img/teoric_limit.svg?sanitize=true)

My recommendation is that you do not hesitate to use it when we have scenes of 500 objects or less and that when the number of objects is greater than 500 you should do a test of whether it is really worth it for your specific case.
