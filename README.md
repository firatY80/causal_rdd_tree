causal_ml.py is a module containing (for now) only one class, which implements a regression tree to discover treatment effect heterogeneity. 
The class implements Reguly (2021) ["Heterogeneous treatment effects in regression discontinuity designs"](https://arxiv.org/abs/2106.11640). Treatment effects within leaves are 
estimated through regression discontinuity, and should thus be interpreted as local effects. Both treatment effects and their variance are 
estimated in an 'honest' fashion as in [Athey and Imbens](https://www.pnas.org/doi/10.1073/pnas.1510489113). 

See [documentation](https://www.firat-yaman.de/causal_tree/index.html) for details and for an example of how to use the CausalTree class.
