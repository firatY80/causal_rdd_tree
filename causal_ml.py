"""
This module only contains one class (CausalTree). It builds a causal tree based 
on Reguly (2021). Treatment effects are local treatment effects based on a 
regression discontinuity estimation within the leaves of the tree.

Usage Steps:
------------

1. Download the ``causal_ml`` module into your working directory.
2. Split your data into training and estimation samples.
3. Create a ``CausalTree`` object.
4. Grow the tree (both training and estimation samples are used in the process).
5. Prune the tree.
6. Estimate unbiased treatment effects with the estimation split.
7. Print the tree and return the leaf information.

Example:
--------

Your main script may look like this:

.. code-block:: python

    import pandas as pd
    from causal_ml import CausalTree
    from sklearn.model_selection import train_test_split

    # Load data
    data = pd.read_csv('your_path_to_the_data')

    # Split data into training and estimation sets
    d_train, d_est = train_test_split(data, test_size=0.5, random_state=42)

    # Initialize CausalTree
    tree = CausalTree(split_steps=20, max_depth=4, min_leaf_size=100)

    # Grow the tree
    tree.grow_tree(d_train, d_est, 'wage', 'time', {'age': 'continuous', 'education': 'discrete'})

    # Prune the tree
    pruned_tree = tree.prune_tree(d_train, d_est, 3)

    # Estimate treatment effects
    pruned_tree.estimate_tree(d_est)

    # Print tree and retrieve leaf information
    leaves = pruned_tree.print_tree()
    
"""
import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class CausalTree:
    """A class for building and pruning regression trees for heterogeneous treatment effects

    Treatment effects are estimated through Regression Discontinuity Design (RDD).
    See Reguly (2021) 'Heterogeneous Treatment Effects in Regression Discontinuity Design'

    :ivar depth: The depth of this node from the source node
    :ivar max_depth: The maximum permissible depth of the tree
    :ivar is_leaf: Whether the node is a leaf
    :ivar left: The node branching to the left
    :ivar right: The node branching to the right
    :ivar tau: The treatment effect in this node, estimated during training
    :ivar tau_est: The treatment effect in this node, estimated on estimation sample
    :ivar v: Variance of tau_est
    """

    def __init__(self, depth=0, max_depth=2, split_steps=10, min_leaf_size=50, 
                 tol=0.005, alpha=0.0):

        if (not isinstance(max_depth, int) or not isinstance(split_steps, int) 
            or not isinstance(min_leaf_size, int)):
            raise TypeError("max_depth, split_steps, and min_leaf_size must be integers.")
        if not isinstance(alpha, (int, float)) or not isinstance(tol, (int, float)):
            raise TypeError("tol and alpha must be of type int or float.")
        if not ((tol>=0.0) and (alpha>=0.0)):
            raise ValueError("tol and alpha must be non-negative.")

        self.depth = depth
        self.tol = tol
        self.split_steps = split_steps
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.alpha = alpha

        self.too_few_samples = False  # too few samples to run RDD
        self.is_leaf = True
        self.left = None
        self.right = None
        # The information gain obtained by splitting this node
        self.delta_error = 0.0 

        # Attributes set during tree growth
        self.tau = 0.0  # Treatment effect 
        self.v = 0.0  # Variance of treatment effect
        self.emse = 0.0  # information value of this node
        
        self.split_var = None
        # spacing for candidates on splitting variable, list
        self.step_size = 0.0
        # lower bounds of splitting variables, list
        self.s_min = None
        # upper bounds of splitting variables, list
        self.s_max = None

    def grow_tree(self, train_data: pd.DataFrame, est_data: pd.DataFrame, dep_var: str, run_var: str, 
                  split_var: list, indep_var: list=[], poly: int=1, cutoff: float=0.0):
        """Grows the full tree

        Recursively splits the training sample until the maximum tree depth is reached, 
        or until there are no further information gains from growing the tree deeper 

        :param train_data: The training sample
        :type train_data: pandas.DataFrame
        :param est_data: The estimation sample
        :type est_data: pandas.DataFrame
        :param dep_var: Name of the dependent variable
        :type dep_var: str
        :param run_var: Name of the running variable
        :type run_var: str
        :param split_var: Names of the splitting variables
        :type split_var: dict
        :param indep_var: Names of the independent variables, defaults to []
        :type indep_var: list, optional
        :param poly: Polynomial order for running variable, defaults to 1
        :type poly: int, optional
        :param cutoff: Cutoff value for running variable, defaults to 0
        :type cutoff: int, optional

        >>> my_tree.grow_tree(data_training, data_estimation, dep_var='wage', run_var='time', 
                            split_var={'age':'continuous', 'education'='discrete'})

        .. note::
           For **run_var**, pass each variable as a key-value pair where 
           key is the variable name (str) and value is 
           either 'discrete' or 'continuous' (str).
        """

        # check that all inputs are in order
        self._validate_inputs(train_data, est_data, dep_var, run_var, 
                              split_var, indep_var, poly)
        
        # drop redundant features, center running variable, and create 
        # polynomial terms if needed
        train_data = self._prep_data(train_data, dep_var, run_var, split_var, 
                                     indep_var, poly, cutoff)
        est_data = self._prep_data(est_data, dep_var, run_var, split_var, 
                                   indep_var, poly, cutoff)
        
        # initialise some tree attributes and obtain emse for this node
        self._initialise_tree(train_data, est_data, dep_var, run_var, 
                              split_var, indep_var, poly, cutoff)
        
        # grow the tree by recursively splitting training and estimation 
        # data
        self._recursive_split(train_data, est_data, self.step_size)

    def _validate_inputs(self, train_data, est_data, dep_var, run_var, 
                         split_var, indep_var, poly):
        if not isinstance(train_data, pd.DataFrame) or not isinstance(est_data, pd.DataFrame):
            raise TypeError("train_data and est_data must be pandas DataFrames.")
        for var in ([dep_var, run_var] + list(split_var) + list(indep_var)):
            if var not in train_data.columns:
                raise ValueError(f"'{var}' not found in training data features.")
        if list(train_data.columns) != list(est_data.columns):
            raise ValueError("Training and estimation data must have the same features.")
        if not isinstance(poly, int) or (poly<1):
            raise ValueError("poly needs to be a positive integer.")

    def _prep_data(self, data, dep_var, run_var, split_var, indep_var, poly, cutoff):
    
        # filter out any unused data 
        data = data[[dep_var, run_var] + list(split_var) + list(indep_var)]
        
        # center running variable at cutoff zero
        data.loc[:,run_var] = data[run_var]-cutoff

        # add polynomial terms for running variable to the data
        for x in range(2, poly+1, 1):
            data[run_var + str(x)] = data[run_var]**x
        
        return data

    def _initialise_tree(self, train_data, est_data, dep_var, run_var, split_var, indep_var, poly, cutoff):
        self.poly = poly
        self.cutoff = cutoff
        self.indep_var = indep_var
        self.split_var = split_var
        self.split_var_names = list(split_var)
        self.dep_var = dep_var
        self.run_var = run_var
        
        # lower bounds of splitting variables
        self.s_min = train_data[self.split_var_names].min()
        # upper bounds of splitting variables
        self.s_max = train_data[self.split_var_names].max()
        
        self.n_train = len(train_data)
        self.n_est = len(est_data)
        self.n_treated = (train_data[run_var] >= 0.0).sum()
        self.n_control = self.n_train - self.n_treated

        # too few observations to run regression
        if min(self.n_treated, self.n_control) < (self.min_leaf_size // 2):
            self.too_few_samples = True
            return
        
        # get spacing between split candidate values
        if self.depth==0:
            self.step_size = (self.s_max-self.s_min)/self.split_steps

        # Compute information value (expected MSE) of this node. 
        # This is EMSE in Reguly paper
        p_plus, p_minus = self._get_p(est_data)
        self.tau, sigma_plus, sigma_minus, m_plus, m_minus = self._get_sigma(train_data)
        v_plus = sigma_plus*m_plus / p_plus
        v_minus = sigma_minus*m_minus / p_minus
        self.v = v_plus + v_minus
        self.emse = self.v * (1.0 / self.n_train + 1.0 / self.n_est)
        self.emse = self.emse - self.tau**2

    def _recursive_split(self, train_data, est_data, step_size):
        if self.max_depth <= self.depth:
            # Tree has already reached its maximum depth
            return
        
        best_emse = self.emse
        best_split = None

        # loop over the splitting variables
        for key, val in self.split_var.items():

            # get the splitting value candidates
            if val=='continuous':
                split_candidates = np.arange(self.s_min[key], self.s_max[key], step_size[key])[1:]
            elif val=='discrete':
                unique_values = sorted(train_data[key].unique())
                split_candidates = unique_values[:-1]
            else:
                raise ValueError('Splitting variable needs to tagged as continous or as discrete')
            
            if len(split_candidates)==0: continue

            # loop over splitting candidates
            for split_val in split_candidates:
                d_left_tr = train_data[train_data[key] <= split_val]
                d_right_tr = train_data[train_data[key] > split_val]
                d_left_est = est_data[est_data[key] <= split_val]
                d_right_est = est_data[est_data[key] > split_val]

                # Too few samples in candidate leaf, try the next split candidate
                if min(len(d_left_tr), len(d_right_tr)) < self.min_leaf_size:
                    continue

                # split into left and right
                treeLeft = CausalTree(depth=self.depth+1, max_depth=self.max_depth, 
                                      tol=self.tol, split_steps=self.split_steps)
                treeRight = CausalTree(depth=self.depth+1, max_depth=self.max_depth, 
                                       tol=self.tol, split_steps=self.split_steps)
                treeLeft._initialise_tree(d_left_tr, d_left_est, self.dep_var, self.run_var, 
                                          self.split_var, self.indep_var, self.poly, self.cutoff)
                treeRight._initialise_tree(d_right_tr, d_right_est, self.dep_var, self.run_var, 
                                           self.split_var, self.indep_var, self.poly, self.cutoff)

                # Too few control or treatment samples in candidate leaves, 
                # try the next split candidate
                if treeLeft.too_few_samples or treeRight.too_few_samples:
                    continue

                tauLeft = (treeLeft.tau**2)*(treeLeft.n_train / self.n_train)
                tauRight = (treeRight.tau**2)*(treeRight.n_train / self.n_train)
                varLeaf = (treeLeft.v + treeRight.v) * (1.0 / self.n_train + 1.0 / self.n_est)
                # expected emse over the new two leaves
                emse_split = varLeaf - tauRight - tauLeft

                # is this the best splitting candidate?
                if emse_split < best_emse:
                    data_left_tr = d_left_tr 
                    data_right_tr = d_right_tr
                    data_left_est = d_left_est
                    data_right_est = d_right_est
                    best_split = (treeLeft, treeRight, split_val)
                    best_emse = emse_split

            # if the best splitting candidate improves the current node EMSE, 
            # then perform split
            if best_split and (best_emse+self.alpha) < (self.emse - self.tol*abs(self.emse)):
                self.is_leaf = False
                # information gain of splitting
                self.delta_error = self.emse - best_emse 
                self.left, self.right, split_val = best_split
                self.left._recursive_split(data_left_tr, data_left_est, step_size)
                self.right._recursive_split(data_right_tr, data_right_est, step_size)

    def _get_sigma(self, data):
        # Helper function to compute some of the components to compute EMSE

        y = data[self.dep_var].values
        vars_to_drop = [self.dep_var] + list(self.split_var_names)
        x = data.drop(columns = vars_to_drop).values
        k = x.shape[1]  # number of independent vars in regression
        treated = data[self.run_var] >= 0.0

        n_total = len(treated)
        n_treated = treated.sum()
        n_control = n_total-n_treated
        if self.too_few_samples:
            print("Warning: There are few control or treatment samples to run the RDD regression.")
        yt = y[treated]   # y vector for treated observations
        yc = y[~treated]  # y vector for control observations
        xt = (x[treated]).reshape(-1,k)
        xc = (x[~treated]).reshape(-1,k)
        rdd_model = LinearRegression()

        rdd_model.fit(xt, yt) 
        alpha_t = rdd_model.intercept_
        yhat = rdd_model.predict(xt)
        res = (yt - yhat)**2
        sigma_plus = res.sum() / (n_treated - k - 1)

        rdd_model.fit(xc, yc) 
        alpha_c = rdd_model.intercept_

        # The treatment effect (difference of the intercepts)
        tau = alpha_t-alpha_c

        yhat = rdd_model.predict(xc)
        res = (yc - yhat)**2
        sigma_minus = res.sum() / (n_control - k - 1)

        m01 = xt.sum()
        m11 = (xt**2).sum()
        m_plus = np.array([[n_treated, m01],[m01, m11]])
        m_plus = np.linalg.inv(m_plus)

        m01 = xc.sum()
        m11 = (xc**2).sum()
        m_minus = np.array([[n_control, m01],[m01, m11]])
        m_minus = np.linalg.inv(m_minus)

        return tau, sigma_plus, sigma_minus, m_plus[0,0], m_minus[0,0]

    def _get_p(self, data):
        # Helper function to get the treatment and control shares 
        # in the estimation sample
        lower_bounds = (data[self.split_var_names] >= self.s_min).all(axis=1)
        upper_bounds = (data[self.split_var_names] <= self.s_max).all(axis=1)
        p_plus = len(data[(data[self.run_var]>=0) & lower_bounds & upper_bounds]) 
        p_minus = len(data[(data[self.run_var]<0) & lower_bounds & upper_bounds])
        p_total = p_plus+p_minus
        return p_plus / p_total, p_minus / p_total

    def _sum_of_leaves(self, train_data, est_data):
        # Helper function to compute the EMSE of the tree

        lower_bounds = (train_data[self.split_var_names] >= self.s_min).all(axis=1)
        upper_bounds = (train_data[self.split_var_names] <= self.s_max).all(axis=1)
        
        # If this node is a leaf, return its contribution to EMSE
        if self.is_leaf:
            d = train_data[lower_bounds & upper_bounds]
            n_te = len(train_data)
            n_est = len(est_data)
            share_leaf = len(d) / n_te
            tau, s1, s2, m1, m2 = self._get_sigma(d)
            p1, p2 = self._get_p(est_data)
            val = (s1*m1/p1 + s2*m2/p2)*(1.0/n_te + 1.0/n_est)
            val -= share_leaf* (tau**2)
            return val
        # If not a leaf, sum the leaves of its children
        total = 0
        if self.left is not None:
            total += self.left._sum_of_leaves(train_data, est_data)
        if self.right is not None:
            total += self.right._sum_of_leaves(train_data, est_data)
        return total

    def print_tree(self):
        """
        Prints out all nodes of the tree along with some of their attributes. 
        Returns a list of the tree leaves if they have attribute ``tau_est``.

        Will print all tree nodes, their depth, boundaries, and estimated treatment
        effects. It will also return a list of leaves, if those leaves already have
        an unbiased treatment estimate (``tau_est``) attached to it.

        :return: Tree leaves with treatment effect, variance, and boundaries for each splitting variable
        :rtype: list or None

        >>> my_tree.print_tree()
        [[0.100, 0.050, [2.1, 2.9], [100, 167]],
         [0.120, 0.081, [2.1, 2.9], [168, 250]]]
        """

        taus = []
        def get_taus(tr):
            borders = np.concatenate((tr.s_min.values, tr.s_max.values)).reshape(2,-1)
            nonlocal taus
            print('')
            print('Depth: ', tr.depth)
            b_list = []
            for id, varname in enumerate(self.split_var_names):
                print(f'Borders {varname} : ', list(borders[:, id]))
                b_list.append(list(borders[:, id]))
            print('Treatment effect (biased): ', tr.tau)
            if tr.is_leaf:
                print('This is a leaf.')
                if hasattr(tr, 'tau_est'):
                    print('Treatment effect (unbiased):', tr.tau_est)
                    taus.append([tr.tau_est, tr.v_est] + b_list)
            if tr.left:
                get_taus(tr.left)
            if tr.right:
                get_taus(tr.right)
        get_taus(self)
        return taus
    
    def estimate_tree(self, data):
        """Estimates unbiased treatment effects in the leaves 

        This method estimates the treatment effect in each leaf of the tree.  
        If you use the estimation sample (as you should), then the treatment
        effect estimates will be unbiased. These estimates are attached to 
        the tau_est attribute.

        :param data: The estimation sample
        :type data: pandas.DataFrame

        >>> my_tree.estimate_tree(data_estimation)

        """                

        est_data = self._prep_data(data, self.dep_var, self.run_var, self.split_var, 
                                   self.indep_var, self.poly, self.cutoff)
            
        if self.is_leaf:
            lower_bounds = (est_data[self.split_var_names] >= self.s_min).all(axis=1)
            upper_bounds = (est_data[self.split_var_names] <= self.s_max).all(axis=1)
            est_data = est_data[lower_bounds & upper_bounds]
            self.tau_est, sigma_plus, sigma_minus, m_plus, m_minus = self._get_sigma(est_data)
            self.v_est = sigma_plus*m_plus + sigma_minus*m_minus
        else:
            self.left.estimate_tree(data)
            self.right.estimate_tree(data)

    def prune_tree(self, train_data, est_data, cv_folds=5):
        """Prune tree using complex-cost pruning

        Uses k-fold cross-validation to prune the tree using complexity-cost pruning

        :param train_data: The training sample
        :type train_data: pandas.DataFrame
        :param est_data: The estimation sample
        :type est_data: pandas.DataFrame
        :param cv_folds: Number of folds for cross-validation, defaults to 5
        :type cv_folds: int, optional

        :return: The pruned tree
        :rtype: CausalTree

        >>> my_tree.prune_tree(data_training, data_estimation)
        <causal_ml.CausalTree at 0x1ff3c9984a0>

        """        

        def _get_alphas(tree, alphas):
            if tree.delta_error not in alphas:
                alphas.append(tree.delta_error)
            if not tree.is_leaf:
                alphas = _get_alphas(tree.left, alphas)
                alphas = _get_alphas(tree.right, alphas)
            return alphas
    
        def _make_folds():
            folds_tr = np.arange(self.n_train) % cv_folds
            folds_est = np.arange(self.n_est) % cv_folds
            np.random.shuffle(folds_tr)
            np.random.shuffle(folds_est)
            return folds_tr, folds_est

        def _get_cv(alphas, folds_tr, folds_est):
            best_alpha = alphas[0]
            best_value = math.inf
            for a in alphas:
                cv_val = 0.0
                for f in range(cv_folds):
                    tree_f = CausalTree(self.depth, self.max_depth, self.split_steps, 
                        self.min_leaf_size, self.tol, alpha = a)
                    selector_tr = folds_tr==f
                    selector_est = folds_est==f
                    tree_f.grow_tree(train_data[~selector_tr], est_data[~selector_est], 
                                     self.dep_var, self.run_var, self.split_var, 
                                     self.indep_var, self.poly, self.cutoff)
                    cv_val += tree_f._sum_of_leaves(train_data[selector_tr], 
                                                    est_data[selector_est])
                if cv_val < best_value:
                    best_alpha = a
                    best_value = cv_val
            return best_alpha
        train_data = self._prep_data(train_data, self.dep_var, self.run_var, self.split_var,
                                     self.indep_var, self.poly, self.cutoff)
        est_data = self._prep_data(est_data, self.dep_var, self.run_var, self.split_var,
                                     self.indep_var, self.poly, self.cutoff)
        alphas = _get_alphas(self, [])
        folds_tr, folds_est = _make_folds()
        alpha_star = _get_cv(alphas, folds_tr, folds_est)

        tree_f = CausalTree(self.depth, self.max_depth, self.split_steps,
                            self.min_leaf_size, self.tol, alpha = alpha_star)
        tree_f.grow_tree(train_data, est_data, self.dep_var, self.run_var, self.split_var, 
                         self.indep_var, self.poly, self.cutoff)
        return tree_f
