# Description

A repository that cintaints the source code to synthesize regret minimizing startegy for the robot operating in a dynamic environment modelled as atwo-player game.

**Table of Contents**
<!-- * [About](https://github.com/MuvvalaKaran/Adam-Can-Play-Any-Strategy/blob/master/README.md#about) -->
* [Create the env](https://github.com/MuvvalaKaran/Adam-Can-Play-Any-Strategy/blob/master/README.md#instructions-to-create-the-env-for-the-code)
* [Project Hierarchy and Implementation](https://github.com/MuvvalaKaran/Adam-Can-Play-Any-Strategy/blob/master/README.md#project-hierarchy-and-implementation)
* [Running the code](https://github.com/MuvvalaKaran/Adam-Can-Play-Any-Strategy/blob/master/README.md#running-the-code)
* [Results](https://github.com/MuvvalaKaran/Adam-Can-Play-Any-Strategy/blob/master/README.md#results)
* [References](https://github.com/MuvvalaKaran/Adam-Can-Play-Any-Strategy/blob/master/README.md#references)

<!-- ## About

This repo contains the code relevant to the theory for variant 1 in [1]. The peseudocode can be found [here](https://github.com/MuvvalaKaran/Adam-Can-Play-Any-Strategy/blob/master/pseudo_code/Variant_I_Pseudocode.pdf). The code can be decoupled into 3 Sub Algorithms :

#### Construction of the function `W'`

`W'` is a set of values representing the `cVal` for each edge `e` in `G` that originates from `Eve's` node such that if `Eve` plays an alternate startegy i.e an alternate edge (if any), then in this alternative play `Adam` plays co-operatively in order to maximize the payoff for `eve`. This value represents the best value `Eve` could receive if she had played the alternate strategy.

#### Construction of the anatagonistic game `G_hat` given `G` the original graph

We then use the values in `W'` say `b` and construct `G_b`(s) with a new weight function `w_hat = w(e) -b` to construct `G_hat`- a collection of `G_b`(s). An edge in `G_b` exists if the corresponfing `W'` value for that edge is less than equal to `b`. If no out-going edges exists then we manually add an edge from the node to `v_terminal`.

#### Computing the Regret value on `G_hat`

Following the claim in *section 3 Claim 1*, it is sufficient to play a memoryless/positional strategies for either players (`Eve` and `Adam`) to ensure an antagonistic value of at least(resp. at most) `aVal(G_hat)`. Thus the regret `Reg(G)` is equal to `-1*aVal(G_hat)`.

There exists a one-to-one correspndance between strategies played on `G_hat` and `G` *if the strategies enter a copy of `G_b` and do not end up in `v_terminal`*. We then map these back to `G` to interpret what a regret minimizing strategy looks like. -->

## Instructions to create the env for the code

* install [`anaconda`](https://www.anaconda.com/products/individual) or [`miniconda`](https://docs.conda.io/en/latest/miniconda.html)

* install [`spot`](https://spot.lrde.epita.fr/install.html) if you are going to construct a DFA using an LTL formula.

* clone this repo with:
 ```bash
git clone --recurse-submodules git@github.com:aria-systems-group/PDDLtoSim.git .
 ```

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

## Running the code

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

Please refere to the README file in the tar ball or on their github [page](https://gitlab.lrde.epita.fr/spot/spot/-/blob/next/README) for more datails on trouble shooting and installation.


## Results

**Stay tuned for updates**

## Contact

Please contact me if you have questions at :karan.muvvala@colorado.edu
