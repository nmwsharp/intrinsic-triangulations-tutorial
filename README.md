# Geometry Processing with Intrinsic Triangulations

Intrinsic triangulations are a powerful technique for computing with 3D surfaces. Among other things, they enable existing algorithms to work "out of the box" on poor-quality triangulations. The basic idea is to represent the geometry of a triangle mesh by edge lengths, rather than vertex positions; this change of perspective unlocks many powerful algorithms with excellent robustness to poor-quality triangulations. 

This course gives an overview of intrinsic triangulations and their use in geometry processing, beginning with a general introduction to the basics and historical roots, then covering recent data structures for encoding intrinsic triangulations, and their application to tasks in surface geometry ranging from geodesics to vector fields to parameterization.

This course was presented at SIGGRAPH 2021 and IMR 2021.

- **Course Notes**: [(pdf link)](https://nmwsharp.com/media/papers/int-tri-course/int_tri_course.pdf)
- **Course Video**: _coming soon_
- **Authors**: [Nicholas Sharp](https://nmwsharp.com/), [Mark Gillespie](https://markjgillespie.com/), [Keenan Crane](http://keenan.is/here)


## Code Tutorial

We provide an implementation of *intrinsic triangulations* from scratch in Python 3, alongside Lawson's algorithm for flipping to an (intrinsic) Delaunay triangulation.
Using these intrinsic triangulations, we compute geodesic distance via the the [heat method](http://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/paper.pdf).

![Screenshot](http://www.cs.cmu.edu/~mgillesp/IntTriCourse/img_small/ScreenshotDropshadow.png)

Install dependencies:
```
python -m pip install numpy scipy polyscope potpourri3d  
```
(`python` might be `python3`, depending on your environment)

Like most PDE-based methods, the heat method may yield inaccurate solutions on low-quality inputs. Running it on a mesh's intrinsic Delaunay triangulation yields dramatically more accurate solutions.
| Mesh        | Distance on Original Mesh           | Distance on Intrinsic Delaunay Triangulation  |
| ------------- |:-------------:| -----:|
| `terrain8k`     | ![Terrain8kBadDistances](http://www.cs.cmu.edu/~mgillesp/IntTriCourse/img_small/terrain8k_input.png) | ![Terrain8kBadDistances](http://www.cs.cmu.edu/~mgillesp/IntTriCourse/img_small/terrain8k_idt.png) |
| `pegasus`     | ![Terrain8kBadDistances](http://www.cs.cmu.edu/~mgillesp/IntTriCourse/img_small/pegasus_input.png) | ![Terrain8kBadDistances](http://www.cs.cmu.edu/~mgillesp/IntTriCourse/img_small/pegasus_idt.png) |
| `rocketship`     | ![Terrain8kBadDistances](http://www.cs.cmu.edu/~mgillesp/IntTriCourse/img_small/rocketship_input.png) | ![Terrain8kBadDistances](http://www.cs.cmu.edu/~mgillesp/IntTriCourse/img_small/rocketship_idt.png) |
