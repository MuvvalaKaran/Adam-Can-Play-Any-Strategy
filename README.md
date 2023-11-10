# Description

This repository contains code for the paper "Let's Collaborate: Regret-based Reactive Synthesis for Robotic Manipulation (ICRA 22)." In this work we introduce algorithms for quantitative regret-based reactive synthesis. We consider resource constrained robotic manipulators that need to interact with a human to achieve a complex task expressed in linear temporal logic. This repository contains the source code to synthesize regret minimizing strategy for the robot operating in a dynamic environment modeled as a two-player turn-based zero-sum game. See the [paper](https://ieeexplore.ieee.org/document/9812298) for more details.

**Table of Contents**
* [Installation](https://github.com/aria-systems-group/regret_synthesis_toolbox/blob/master/README.md#Installation)
* [Results](https://github.com/aria-systems-group/regret_synthesis_toolbox/blob/master/README.md#results)
* [Reference](https://github.com/aria-systems-group/regret_synthesis_toolbox/blob/master/README.md#Citing)


# Installation

## Clone the code

* clone this repo with:
 ```bash
git clone --recurse-submodules git@github.com:aria-systems-group/regret_synthesis_toolbox.git .
 ```

Note: The `--recurse-submodule` will automatically initialize and update each submodule in the repository. This repository contains following synthesis algorithms.

### Supported:

Qualitative Algorithms

- [x] `Adversarial Game (w Permissive strategy synthesis)`
- [x] `Cooperative Game (w Permissive strategy synthesis)`
- [ ]  `Best-Effort Synthesis`([Link](https://www.ijcai.org/proceedings/2021/243))

Quantitative Algorithms

- [x] `Value Iteration (Min-Max and Min-Min) w total Payoff function`([Link](https://link.springer.com/article/10.1007/s00236-016-0276-z))
- [x] `Finite Trace Regret Synthesis`([Link](https://muvvalakaran.github.io/publication/icra_22_regret/))
- [x] `Bounded Human Adversarial Game`([Link](https://ieeexplore.ieee.org/abstract/document/8206426))
- [ ] `Best-Effort Synthesis`

For regret-synthesis abstraction and synthesis see the [PDDLtoSim](https://github.com/aria-systems-group/PDDLtoSim) repository.

## Docker Installation - Creating an Image and Spinning a Container

Make sure you have Docker installed. If not, follow the instructions [here](https://docs.docker.com/engine/install/ubuntu/).

### Docker Commands to build the image

1. `cd` into the root of the project

2. Build the image from the Dockerfile

```bash
docker build -t <image_name> .
```

Note: the dot looks for a Dockerfile in the current repository. Then spin an instance of the container by using the following command

```bash
docker run -it --name <docker_container_name> <docker image name>
```

For volume binding

```bash
docker run -v <HOST-PATH>:<Container-path>
```

For example, to volume bind your local directory to the `regret_planning` folder inside the Docker, use the following command

```bash
docker run -it -v $PWD:/root/regret_planning --name <dokcer_container_name> <image_name>
```

Here `<docker_container_name>` is any name of your choice and `<image_name>` is the docker image name from above. `-it` and `-v` are flags to run an interactive terminal and volume bind respectively. 

Additionally, if you are more used to GUI and would like to edit or attach a container instance to VSCode ([Link](https://code.visualstudio.com/docs/devcontainers/containers)) then follow the instructions below:

### Attaching the remote container to VScode


1. Make sure you have the right VS code extensions installed
   * install docker extension
   * install python extension
   * install remote container extension
   * Now click on the `Remote Explore` tab on the left and attach VScode to a container.
2. This will launch a new vs code attached to the container and prompt you to a location to attach to. The default is root, and you can just press enter. Congrats, you have attached your container to VSCode.

### Running the code

`cd` into the `regret_planning` directory, and run the following command

```bash
python3 main.py
```


## Conda Installation - Instructions to create the env for the code

* install [`anaconda`](https://www.anaconda.com/products/individual) or [`miniconda`](https://docs.conda.io/en/latest/miniconda.html)

* install [`spot`](https://spot.lrde.epita.fr/install.html) if you are going to construct a DFA using an LTL formula.

* change into this repo's directory:
 ```bash
cd regret_synthesis_toolbox
 ```
* create the `conda` environment for this library:
```bash
conda env create -f environment.yml
 ```

* activate the conda environment:
 ```bash
conda activate regret_syn_env
 ```

### Running the code

`cd` into the root directory, activate the conda `env`  and run the following command

```bash
python3 main.py
```

## Spot Troubleshooting notes

You can build `spot` from source, official git [repo](https://gitlab.lrde.epita.fr/spot/spot) or Debain package. If you do source intallation, then run the following command to verify your installation

```bash
ltl2tgba --version

```

If your shell reports that ltl2tgba is not found, add `$prefix/bin` to you $PATH environment variable by using the following command

```bash
export PATH=$PATH:/place/with/the/file

```

Spot installs five types of files, in different locations. $prefix refers to the directory that was selected using the --prefix option of configure (the default is /usr/local/).

1) command-line tools go into $prefix/bin/
2) shared or static libraries (depending on configure options)
   are installed into $prefix/lib/
3) Python bindings (if not disabled with --disable-python) typically
   go into a directory like $prefix/lib/pythonX.Y/site-packages/
   where X.Y is the version of Python found during configure.
4) man pages go into $prefix/man
5) header files go into $prefix/include

Please refer to the README file in the tar ball or on their Github [page](https://gitlab.lrde.epita.fr/spot/spot/-/blob/next/README) for more details on trouble shooting and installation.

# Results

You can find more information at this project [link](https://muvvalakaran.github.io/publication/icra_22_regret/)


# Citing

If the code is useful in your research, and you would like to acknowledge it, please cite this [paper](https://ieeexplore.ieee.org/document/9812298):

```
@INPROCEEDINGS{muvvala2022regret,
  author={Muvvala, Karan and Amorese, Peter and Lahijanian, Morteza},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)}, 
  title={Let's Collaborate: Regret-based Reactive Synthesis for Robotic Manipulation}, 
  year={2022},
  pages={4340-4346},
  doi={10.1109/ICRA46639.2022.9812298}}
```

# Contact

Please contact me if you have questions at :karan.muvvala@colorado.edu
