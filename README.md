# Description

A project implementing the first variant of regret minimization strategy for `Eve`(the controller/system) when `Adam`(the human/env) can play any strategy. More information regarding the theory can be found in [1].

**Table of Contents**
* [About](https://github.com/MuvvalaKaran/Adam-Can-Play-Any-Strategy/blob/master/README.md#about)
* [Create the env](https://github.com/MuvvalaKaran/Adam-Can-Play-Any-Strategy/blob/master/README.md#instructions-to-create-the-env-for-the-code)
* [Project Hierarchy and Implementation](https://github.com/MuvvalaKaran/Adam-Can-Play-Any-Strategy/blob/master/README.md#project-hierarchy-and-implementation)
* [Running the code](https://github.com/MuvvalaKaran/Adam-Can-Play-Any-Strategy/blob/master/README.md#running-the-code)
* [Results](https://github.com/MuvvalaKaran/Adam-Can-Play-Any-Strategy/blob/master/README.md#results)
* [References](https://github.com/MuvvalaKaran/Adam-Can-Play-Any-Strategy/blob/master/README.md#references)

## About

This repo contains the code relevant to the theory for variant 1 in [1]. The peseudocode can be found [here](https://github.com/MuvvalaKaran/Adam-Can-Play-Any-Strategy/blob/master/pseudo_code/Variant_I_Pseudocode.pdf). The code can be decoupled into 3 Sub Algorithms :

#### Construction of the function `W'`

`W'` is a set of values representing the `cVal` for each edge `e` in `G` that originates from `Eve's` node such that if `Eve` plays an alternate startegy i.e an alternate edge (if any), then in this alternative play `Adam` plays co-operatively in order to maximize the payoff for `eve`. This value represents the best value `Eve` could receive if she had played the alternate strategy.

#### Construction of the anatagonistic game `G_hat` given `G` the original graph

We then use the values in `W'` say `b` and construct `G_b`(s) with a new weight function `w_hat = w(e) -b` to construct `G_hat`- a collection of `G_b`(s). An edge in `G_b` exists if the corresponfing `W'` value for that edge is less than equal to `b`. If no out-going edges exists then we manually add an edge from the node to `v_terminal`.

#### Computing the Regret value on `G_hat` 

Following the claim in *section 3 Claim 1*, it is sufficient to play a memoryless/positional strategies for either players (`Eve` and `Adam`) to ensure an antagonistic value of at least(resp. at most) `aVal(G_hat)`. Thus the regret `Reg(G)` is equal to `-1*aVal(G_hat)`. 

There exists a one-to-one correspndance between strategies played on `G_hat` and `G` *if the strategies enter a copy of `G_b` and do not end up in `v_terminal`*. We then map these back to `G` to interpret what a regret minimizing strategy looks like. 

## Instructions to create the env for the code

1. **Installing the env**\
Install anaconda on your os and then run the below moentioned shell commands with conda path

2. Install dependencies - Install conda env

    Use the terminal or an Anaconda Prompt for the following steps:

* Create the environment from the environment.yaml file under conda

    ```bash
    conda env create -f variant_1.yml
    ```

* The first line of the yml file sets the new environment's name. In my case it is `adam_can_play_any_strategy`. You could modify it before compiling this file.

*  Verify that the new environment was installed correctly. You should see a star('\*') in front of the current active env

    ```bash
    conda env list
    ```

    you can also use

    ```bash
    conda info --envs
    ```

* If you don't see a ('\*') in front of the newly created env, but can see it in the env list then activate the env using the following command

	``` bash
	conda activate adam_can_play_any_strategy 
	```

## Project Hierarchy and Implementation

1. main : The main module that contains the code relevant to regret minimization strategy syntheis using Variant 1 Pseudocode. The Pseucode can be found [here](https://github.com/MuvvalaKaran/Adam-Can-Play-Any-Strategy/blob/add_feature_product_grpah_pruning/pseudo_code/Variant_I_Pseudocode.pdf).

	The `main()` method implements the regret minimization pseudocode and calls the following functions. 

	1. `construct_graph()` : A method to call the factory class GraphFactory() to construct a two player game `G` with nodes that bleong to `Eve`(the system/ controller player) and `Adam`(the env/human player). Each edge `e` is initialized with an edge weight given by the weight function `w(e)`. 

	Depending on the function, if we call `construct_two_player_game()` then we build the two player graph with env and system nodes manually else if call the `construct_product_graph()` method with an `LTL formula` the automatically creates this two player graph for you given the following things:

	- A transition system (*should be total*) consisting of only `Eve's` node
	- A scLTL (co-safe LTL) formula eg : `!b U c` 
	- The number of times the human can intervene(`k`)

	Given a transition system such as the one below 

	<p align="center">
		<img src="http://yuml.me/diagram/scruffy/activity/[s1]->[s2]->[s3], [s1]->[s3], [s3]->[s1], [s2]->[s1]">
	</p>

	We add a human node after each edge originating from the system nodes and add a counter which represents how many time the human as intervened e.g `s1,0` mean the human as intervened `0` times. If the human does not take any action then we proceed to the original states. If the human does take an action then that edge is represented with `m` and we increment the counter by 1. Thus we get the transition system(T).

	After constructing T, we compose it with the A (The Deterministic Finite Automaton) - we use co-safe LTL to specify a task as it the ideal framework to express robotic tasks that must be accomplished in finite time. 

	<p align="center">
		<img src="http://yuml.me/diagram/scruffy/activity/[s1,0]-s12>(h12,0),(h12,0)-s12>[s2,0],[s2,0]-s23>(h23,0),(h23,0)-s23>[s3,0],[s1,0]-s13>(h13,0),(h13,0)-s13>[s3,0],[s3,0]-s31>(h31,0),(h31,0)-s31>[s1,0],[s2,0]-s21>(h21,0),(h21,0)-s21>[s1,0],(h12,0)-m>[s1,1],(h12,0)-m>[s3,1],(h23,0)-m>[s1,1],(h23,0)-m>[s2,1],(h13,0)-m>[s2,1],(h13,0)-m>[s1,1],(h31,0)-m>[s2,1],(h31,0)-m>[s3,1],(h21,0)-m>[s2,1], (h21,0)-m>[s3,1],[s1,1]-s12>[s2,1],[s2,1]-s23>[s3,1],[s2,1]-s21>[s1,1],[s1,1]-s13>[s3,1],[s3,1]-s31>[s1,1]">
	</p>
	
	2. `construct_w_prime()` : A method implementing Algorithm 2 of the Pseudocode. This method computes `W'` - a set of cooperative values( `cVal`) for each edge `e = (u, v)` that belong to `G` such that if `Eve` plays an alternate startegy (i.e playing any other edge `e = (u, v')`) then `Adam` plays cooperatively in this alternative strategy played by `Eve`. So `Adam` is trying to maximize `Eve's` payoff for the alternate strategies.
	
	3. `construct_g_hat()` : A method implementing Algorithm 3 of the Pseudocode. This method construct the antagonistic game `G_hat` which is composed of `G_b`(s) where `b` is the value that belongs to the set `W'`. The new weight function `w_hat(.)` is given by `w(e) - b` for `G_b` in `G_hat`.
	
	4. `compute_aVal()` : A method to compute the antagonistic value for the game `G_hat` following Algorithm 4 of the Pseudocode. Regret for the game `G` is equivalent to `-aVal(G_hat)`. This function also returns the memoryless strategy followed by `Eve` and `Adam` on `G_hat`.

	5. `map_g_hat_to_org_graph()` : A helper method to map back the original strategy for `Eve` and `Adam` from `G_hat` to `G`.

2. src/compute_payoff :  A module implementing the class payoff_value used to quantify the value of all the possible infinte loops in a given game with a given initial node. 

3. src/graph/graph.py : A module implementing the abstract base Graph class. This class implements the foundational member function like creating a Multi-Directed Graph from the networkx package, methods to add states, edges, and states and edges with attributes along with method to dump a Graph in a yaml file and read and plot it using graphviz. Below are the child classes :
	<p align="center">	
		<img src="http://yuml.me/diagram/scruffy/class/[Graph]^-[TwoPlayerGraph], [Graph]^-[DFAGraph]" align="center" border="0">
	</p>
	
	1. TwoPlayerGraph : A class that implements the construction of a two player graph. We can use this class to directly construct a two player game and feed it to the regret minimization strategy synthesis code in main() module to compute the least regret strategies for `Eve`. Else we can construct the product graph (`P`) using the `ProductAutomaton` class that takes as input a Transition system(T) and an Automaton (in our case a Deterministic Finite Automaton(DFA) - `A`) and takes the product of these two. 
	
	2. DFAGraph(`A`) : A class that implements the construction of a Buchi Automaton(BA) . This BA is the output [spot](https://spot.lrde.epita.fr/) toolbox which takes in as input a LTL formula(in our case it is a co-safe LTL formula) and output is a BA in NeverClaim format. This output is interpreted using the src/graph/promela.py module. 
	
	The child classes of TwoPlayerGraph are :
	
	<p align="center">
		<img align="center" border="0" src="http://yuml.me/diagram/scruffy/class/[TwoPlayerGraph]^-[GminGraph], [TwoPlayerGraph]^-[GmaxGraph], [TwoPlayerGraph]^-[FiniteTransSys], [TwoPlayerGraph]^-[ProductAutomaton]">
	</p>

	1. GminGraph : A class that implements the construction of a gmin graph for `inf` payoff function following the theory in the paper.

	2. GmaxGraph : A class that implements the construction of a gmax graph for `sup` payoff function following the theory in the paper
	
	3. FiniteTransSys(`T`) : A class that implements the construction of a Finite Transition system graph. There are some option methods in this class that helps build a two player transition system custom to our application and then prune edges that belong to `Eve` that do not eventually lead to the accepting states.  
	
	4. ProductAutomaton(`P`) : A class that automates the construction of a two player graph given T and A. 


4. src/graph/spot.py : A module that implements methods to run spot through a shell and interpret the raw output using an ascii decoder. 

## Running the code 

cd into the root folder and open up a terminal. Activate the conda environment that you created following steps in the [Instructions to create the env for the code](https://github.com/MuvvalaKaran/Adam-Can-Play-Any-Strategy#instructions-to-create-the-env-for-the-code). To actiavte the env type in the following statement:

``` bash
conda activate <name_of_the_env>
```

The default value of `<name_of_the_env>` is `adam_can_play_any_strategy`. After this run `which python` just to make sure that you are indeed using the python from the new created conda env which also appears within `( )` in front of your user-name in the terminal. The output should be similar to this (at least for linux users):

``` bash
path/to/anaconda3/envs/<name_of_the_env>/bin/pythonX
```

Now run the following the main file as follows 

```bash
python ./main.py 
```

The terminal will prompt you to enter either 1 or 2. Enter 1 if you want to use the example used in the paper. Enter 2 if you want to use the custom built transition system

**NOTE: As of right now the code does not ask to enter an LTL formula or ask the user to construct the raw transition system (one with only `Eve` nodes) through terminal. In future updates, this feature will be added.**

## Results

**Stay tuned for updates**

## References

>[1] Hunter, Paul, Guillermo A. Pérez, and Jean-François Raskin. "Reactive synthesis without regret." arXiv preprint arXiv:1504.01708 (2015).

