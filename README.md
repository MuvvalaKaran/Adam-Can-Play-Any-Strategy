# Description

This code is the implementation of the first variant of the regret strategy for Eve(controller) when Adam(env) can play any strategy.

# Instructions to run the code

1. **Installing the env**\
Install anaconda on your os and then run the below moentioned shell commands with conda path

2. Install dependencies - Install conda env

    Use the terminal or an Anaconda Prompt for the following steps:

* Create the environment from the environment.yaml file under conda

    ```conda env create -f variant_1.yml```

* The first line of the yml file sets the new environment's name. In my case it is "". you could modify it before compiling this file.

*  Verify that the new environment was installed correctly. You should see a star('\*') in front of the current active env

    ```conda env list```

    you can also use

    ```conda info --envs```
		
# Running the code 

# Strategy synthesis for eve when Adam can play any strategy

# Result

<!-- ![](src/graph/org_graph.jpg "org_graph") -->
<p align="center">
<img src="src/graph/org_graph.jpg" alt="org_graph" width="500" height="500">
</p>

---
<p align="center">
<!-- ![](src/graph/Gmax_graph.jpg "gmax_graph") -->
<img src="src/graph/Gmax_graph.jpg" alt="gmax_graph" width="500" height="500">
</p>

---
<p align="center">
<!-- ![](src/graph/Gmin_graph.jpg "gmin_graph") -->
<img src="src/graph/Gmin_graph.jpg" alt="gmin_graph" width="500" height="500">
</p>

---
<p align="center">
<!-- ![](src/graph/G_hat.jpg "g_hat_graph") -->
<img src="src/graph/G_hat.jpg" alt="g_hat_graph" width="300" height="600">
</p>

---


# References

`Hunter, Paul, Guillermo A. Pérez, and Jean-François Raskin. "Reactive synthesis without regret." arXiv preprint arXiv:1504.01708 (2015).`

