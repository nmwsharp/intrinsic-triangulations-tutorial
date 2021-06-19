# intrinsic-triangulations-tutorial

We provide an implementation of *intrinsic triangulations* from scratch in python, alongside Lawson's algorithm for flipping to an (intrinsic) Delaunay triangulation.
Using these intrinsic triangulations, we compute geodesic distance via the the [heat method](http://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/paper.pdf).

![Screenshot](http://www.cs.cmu.edu/~mgillesp/IntTriCourse/img_small/ScreenshotDropshadow.png)

Like most PDE-based methods, the heat method may yield inaccurate solutions on low-quality inputs. Running it on a mesh's intrinsic Delaunay triangulation yields dramatically more accurate solutions.
| Mesh        | Distance on Original Mesh           | Distance on Intrinsic Delaunay Triangulation  |
| ------------- |:-------------:| -----:|
| `terrain8k`     | ![Terrain8kBadDistances](http://www.cs.cmu.edu/~mgillesp/IntTriCourse/img_small/terrain8k_input.png) | ![Terrain8kBadDistances](http://www.cs.cmu.edu/~mgillesp/IntTriCourse/img_small/terrain8k_idt.png) |
| `pegasus`     | ![Terrain8kBadDistances](http://www.cs.cmu.edu/~mgillesp/IntTriCourse/img_small/pegasus_input.png) | ![Terrain8kBadDistances](http://www.cs.cmu.edu/~mgillesp/IntTriCourse/img_small/pegasus_idt.png) |
| `rocketship`     | ![Terrain8kBadDistances](http://www.cs.cmu.edu/~mgillesp/IntTriCourse/img_small/rocketship_input.png) | ![Terrain8kBadDistances](http://www.cs.cmu.edu/~mgillesp/IntTriCourse/img_small/rocketship_idt.png) |
