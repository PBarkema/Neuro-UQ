"""Python implementation of a Bayesian probabilistic learner.

This model was originally described in Behrens et al. Nat Neuro 2007.

The code here was adapted from the original C++ code provided by
Tim Behrens.

"""
from __future__ import division
import numpy as np
from numpy import log, exp, power, pi
from scipy.special import gammaln
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    import warnings
    message = "Could not import seaborn; plotting will not work."
    warnings.warn(message, UserWarning)


class ProbabilityLearner(object):

    def __init__(self, p_step=.01, I_step=.1):

        # Set up the parameter grids
        p_grid = make_grid(.01, .99, p_step)
        self.p_grid = p_grid
        self.vol_idx = range(0,5)
        self.vol_val = 0.97
        self.epsilon = 1e-1
        I_grid = make_grid(log(2), log(10000), I_step)
        self.I_grid = I_grid

        self._p_size = p_grid.size
        self._I_size = I_grid.size

        # Set up the transitional distribution on p
        joint_grid = np.meshgrid(p_grid, p_grid, I_grid, indexing="ij")
        p_trans = np.vectorize(p_trans_func)(*joint_grid)
        self._p_trans = p_trans / p_trans.sum(axis=0)

        # Initialize the learner and history
        self.reset()

    @property
    def p_hats(self):
        return np.atleast_1d(self._p_hats)

    @property
    def I_hats(self):
        return np.atleast_1d(self._I_hats)

    @property
    def data(self):
        return np.atleast_1d(self._data)

    def fit(self, data):
        """Fit the model to a sequence of Bernoulli observations."""
        if np.isscalar(data):
            data = [data]
        for y in data:
            self._update(y)
            pI = self.pI
            self.p_dists.append(pI.sum(axis=1))
            self.I_dists.append(pI.sum(axis=0))
            self._p_hats.append(np.sum(self.p_dists[-1] * self.p_grid))
            self._I_hats.append(np.sum(self.I_dists[-1] * self.I_grid))
            self._data.append(y)

    def _update(self, y):
        """Perform the Bayesian update for a trial based on y."""

        # Information leak (increase in the variance of the joint
        # distribution to reflect uncertainty of a new trial)
        # -------------------------------------------------------

        pI = self.pI

        # Multiply P(p_p+1 | p_i, I) by P(p_i, I) and
        # integrate out p_i, which gives P(p_i+1, I)

        # pI = (self._p_trans * pI).sum(axis=1)

        # Equivalent but twice as fast:
        pI = np.einsum("ijk,jk->ik", self._p_trans, pI)

        # Update P(p_i+1, I) based on the newly observed data
        # ----------------------------------------------------------

        likelihood = self.p_grid if y else 1 - self.p_grid
        pI *= likelihood[:, np.newaxis]

        # Normalize the new distribution
        # ------------------------------

        self.pI = pI / pI.sum()

    def reset(self):
        """Reset the history of the learner."""
        # Initialize the joint distribution P(p, I, k)
        epsilon = self.epsilon
        pI = np.ones((self._p_size, self._I_size)) * epsilon
        pI[:, self.vol_idx] += self.vol_val
        self.pI = pI / pI.sum()

        # Initialize the memory lists
        self.p_dists = []
        self.I_dists = []
        self._p_hats = []
        self._I_hats = []
        self._data = []
    def force_reset_new_tone(self):
            """
            Resets beliefs to model a NEW TONE (Context Switch).
            1. Probability (p): Returns to Uniform (Flat).
            2. Volatility (I): Spikes to High Volatility (Low I).
            """
            epsilon = self.epsilon
            # Create a mostly empty grid
             # 1. FLAT P: We don't know what the new tone predicts. 
            # (This is already handled because the grid is uniform in dimension 0)
            pI = np.ones((self._p_size, self._I_size)) * epsilon
             # 2. HIGH VOLATILITY: We are uncertain.
            # Inject mass into the lowest I indices (Index 0 on axis 1)
            pI[:, self.vol_idx] += self.vol_val
           
            # 3. Normalize
            self.pI = pI / pI.sum()
            
            print(">> New Tone Detected: Beliefs Reset to Flat Prior.")
    # def force_reset(self):
    #     """
    #     Resets beliefs with a 'Soft' reset to avoid the Zero Probability Trap.
    #     """
    #     epsilon = self.epsilon # A tiny non-zero probability
    #     # Initialize with epsilon
    #     pI = np.ones((self._p_size, self._I_size)) * epsilon
        
    #     # Add massive spike to High Volatility
    #     pI[:, 0] += self.vol_val
        
    #     # 3. Normalize
    #     self.pI = pI / pI.sum()

        
            # if hasattr(self, 'pIk'): # VolatilityLearner
            #     # 1. Initialize with epsilon (not zeros!)
            #     new_dist = np.ones((self._p_size, self._I_size, self._k_size)) * epsilon
                
            #     # 2. Add a massive spike of probability to the High Volatility region
            #     # Index 0 of axis 1 is the lowest I (highest volatility)
            #     # We make p uniform (all indices of axis 0)
            #     new_dist[:, 0, :] += 1.0 
                
            #     # 3. Normalize
            #     self.pIk = new_dist / new_dist.sum()

    def get_metrics(self, y=None):
        U_hats = [1/u for u in np.exp(self.I_hats)]
        U_hats_scaled = (U_hats - np.min(U_hats)) / (np.max(U_hats) - np.min(U_hats))
        self.U_hats = U_hats_scaled

        PE = (self.p_hats - y)**2
        # Calculation of PE ^ UU as combined uncertainty measure
        PE_Uhat = PE * self.U_hats
        return self.p_hats, self.U_hats, PE_Uhat
    
    def plot_history(self, ground_truth=None, y=None, **kwargs):
        """Plot the data, posterior means, and p*U interaction."""
        
        # # --- 1. Calculation of U (Uncertainty) ---
        # # Inverse of precision = Variance proxy
        # U_hats_raw = np.array([1/u for u in np.exp(self.I_hats)])
        
        # # Min-Max Scale U to 0-1
        # u_min = np.min(U_hats_raw)
        # u_max = np.max(U_hats_raw)
        
        # # Safety check to avoid divide by zero if flat line
        # if u_max - u_min == 0:
        #     self.U_hats = np.zeros_like(U_hats_raw)
        # else:
        #     self.U_hats = (U_hats_raw - u_min) / (u_max - u_min)
        scale_max = np.percentile(U_hats, 90)
        scale_min = np.percentile(U_hats, 10)
        U_hats = [1/u for u in np.exp(self.I_hats)]
        U_hats_scaled = (U_hats - scale_min) / (scale_max -  scale_min)
        self.U_hats = U_hats_scaled

        PE = (self.p_hats - y)**2
        # Calculation of PE ^ UU as combined uncertainty measure
        PE_Uhat = PE * self.U_hats

        # Calculation of Product (p * U) ---
        #product_metric = self.p_hats * self.U_hats

    
        # Get 3 colors for the 3 plots
        blue, green, red = sns.color_palette("deep", n_colors=3)

        trials = np.arange(self.data.size)
        xlim = trials.min(), trials.max()

        # Create 3 rows, sharing the X axis
        f, (p_ax, I_ax, prod_ax) = plt.subplots(3, 1, sharex=True, **kwargs)

        # --- Plot 1: Probability (p_hat) ---
        p_ax.plot(trials, self.p_hats, c=blue)
        p_ax.scatter(trials, self.data, c=".25", alpha=.5, s=15)

        if ground_truth is not None:
            p_ax.plot(trials, [ground_truth for t in trials], c="dimgray", ls="--")
        
        p_ax.set_ylabel("$\hat p$", size=16)
        p_ax.set(xlim=xlim, ylim=(-.1, 1.1))

        # --- Plot 2: Uncertainty (U_hat) ---
        I_ax.plot(trials, self.U_hats, c=green)
        I_ax.set_ylabel("$\hat U$", size=16)
        I_ax.set(ylim=(-0.1, 1.1)) # Scaled 0-1

        # --- Plot 3: Interaction (p * U) ---
        prod_ax.plot(trials, PE_Uhat, c=red)
        prod_ax.set_ylabel(r"update", size=16) #$\hat p ** \hat U$
        prod_ax.set_xlabel("Trial") # Label only the bottom plot
        #prod_ax.set(ylim=(-0.1, 1.1)) # Product of two 0-1 vars is 0-1


        f.tight_layout()

    def plot_joint(self, cmap="BuGn"):
        """Plot the current joint distribution P(p, I)."""
        pal = sns.color_palette(cmap, 256)
        lc = pal[int(.7 * 256)]
        bg = pal[0]
        
        fig = plt.figure(figsize=(7, 7))
        gs = plt.GridSpec(6, 6)
        
        p_lim = self.p_grid.min(), self.p_grid.max()
        I_lim = self.I_grid.min(), self.I_grid.max()
     
        ax1 = fig.add_subplot(gs[1:, :-1])
        ax1.set(xlim=p_lim, ylim=I_lim)

        ax1.contourf(self.p_grid, self.I_grid, self.pI.T, 30, cmap=cmap)

        ax1.set_xlabel("$p$", size=16)
        ax1.set_ylabel("$I$", size=16)

        ax2 = fig.add_subplot(gs[1:, -1], axis_bgcolor=bg)
        ax2.set(ylim=I_lim)
        ax2.plot(self.pI.sum(axis=0), self.I_grid, c=lc, lw=3)
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3 = fig.add_subplot(gs[0, :-1], axis_bgcolor=bg)
        ax3.set(xlim=p_lim)
        ax3.plot(self.p_grid, self.pI.sum(axis=1), c=lc, lw=3)
        ax3.set_xticks([])
        ax3.set_yticks([])

    def show_model(self):
        """Render the model as a Bayes net using daft."""
        import daft
        gray = ".3"
        pgm = daft.PGM((3, 3), node_ec=gray)
        scale = 1.5

        pgm.add_node(daft.Node("v", r"$v$", 1.5, 2.5, scale))
        pgm.add_node(daft.Node("pim1", r"$p_{i-1}$", .5, 1.5, scale))
        pgm.add_node(daft.Node("pi", r"$p_i$", 2.5, 1.5, scale))
        pgm.add_node(daft.Node("yim1", r"$y_{i-1}$", .5, .5, scale, observed=True))
        pgm.add_node(daft.Node("yi", r"$y_i$", 2.5, .5, scale, observed=True))

        kws = {"plot_params": {"ec": gray, "fc": gray}}
        pgm.add_edge("v", "pim1", **kws)
        pgm.add_edge("v", "pi", **kws)
        pgm.add_edge("pim1", "pi", **kws)
        pgm.add_edge("pim1", "yim1", **kws)
        pgm.add_edge("pi", "yi", **kws)

        pgm.render()
        return pgm
    


class VolatilityLearner(ProbabilityLearner):

    def __init__(self, p_step=.02, I_step=.2, k_step=.2):

        # Set up the parameter grids
        self.p_grid = make_grid(.01, .99, p_step)
        self.I_grid = make_grid(log(2), log(10000), I_step)
        self.k_grid = make_grid(log(5e-4), log(20), k_step)

        self._p_size = self.p_grid.size
        self._I_size = self.I_grid.size
        self._k_size = self.k_grid.size

        # Set up the transitional distributions
        I_trans = np.vectorize(I_trans_func)(*np.meshgrid(self.I_grid,
                                                          self.I_grid,
                                                          self.k_grid,
                                                          indexing="ij"))
        self._I_trans = I_trans / I_trans.sum(axis=0)

        p_trans = np.vectorize(p_trans_func)(*np.meshgrid(self.p_grid,
                                                          self.p_grid,
                                                          self.I_grid,
                                                          indexing="ij"))
        self._p_trans = p_trans / p_trans.sum(axis=0)

        # Initialize the learner and history
        self.reset()

    def _update(self, y):
        """Perform the Bayesian update for a trial based on y."""

        # Information leak (increase in the variance of the joint
        # distribution to reflect uncertainty of a new trial)
        # -------------------------------------------------------

        pIk = self.pIk

        # Multiply P(I_i+1 | I_i, k) by P(p_i, I_i, k) and
        # integrate out I_i, which gives P(p_i, I_i+1, k)
        I_leaked = np.einsum("jkl,ikl->ijl", self._I_trans, pIk)

        # Multiply P(p_p+1 | p_i, I_i+1) by P(p_i, I_i+1, k) and
        # integrate out p_i, which gives P(p_i+1, I_i+1, k)
        p_leaked = np.einsum("ijk,jkl->ikl", self._p_trans, I_leaked)

        # Set the running joint distribution to the new values
        pIk = p_leaked

        # Update P(p_i+1, I_i+1, k) based on the newly observed data
        # ----------------------------------------------------------

        likelihood = self.p_grid if y else 1 - self.p_grid
        pIk *= likelihood[:, np.newaxis, np.newaxis]

        # Normalize the new distribution
        # ------------------------------

        self.pIk = pIk / pIk.sum()

    @property
    def k_hats(self):
        return np.atleast_1d(self._k_hats)

    @property
    def pI(self):
        return self.pIk.mean(axis=-1)

    def fit(self, data):
        """Fit the model to a sequence of Bernoulli observations."""
        for y in data:
            self._update(y)
            pI = self.pIk.sum(axis=2)
            self.p_dists.append(pI.sum(axis=1))
            self.I_dists.append(pI.sum(axis=0))
            self.k_dists.append(self.pIk.sum(axis=(0, 1)))
            self._p_hats.append(np.sum(self.p_dists[-1] * self.p_grid))
            self._I_hats.append(np.sum(self.I_dists[-1] * self.I_grid))
            self._k_hats.append(np.sum(self.k_dists[-1] * self.k_grid))
            self._data.append(y)

    def reset(self):
        """Reset the history of the learner."""
        # Initialize the joint distribution P(p, I, k)
        pIk = np.ones((self._p_size, self._I_size, self._k_size))
        self.pIk = pIk / pIk.sum()

        # Initialize the memory lists
        self.p_dists = []
        self.I_dists = []
        self.k_dists = []
        self._p_hats = []
        self._I_hats = []
        self._k_hats = []
        self._data = []

    def plot_history(self, ground_truth=None, **kwargs):
        """Plot the data and posterior means from the history."""
        blue, green, red = sns.color_palette("deep", n_colors=3)

        trials = np.arange(self.data.size)
        xlim = trials.min(), trials.max()

        f, (p_ax, I_ax, k_ax) = plt.subplots(3, 1, sharex=True, **kwargs)
        p_ax.plot(trials, self.p_hats, c=blue)
        p_ax.scatter(trials, self.data, c=".25", alpha=.5, s=15)

        if ground_truth is not None:
            p_ax.plot(trials, ground_truth, c="dimgray", ls="--")
        p_ax.set_ylabel("$\hat p$", size=16)
        p_ax.set(xlim=xlim, ylim=(-.1, 1.1))

        I_ax.plot(trials, self.I_hats, c=green)
        I_ax.set_ylabel("$\hat I$", size=16)
        I_ax.set_ylim(2, 10)

        k_ax.plot(trials, self.k_hats, c=red)
        k_ax.set_ylabel("$\hat k$", size=16)
        k_ax.set(xlabel="Trial", ylim=(-8, 3))
        f.tight_layout()

    def show_model(self):
        """Render the model as a Bayes net using daft."""
        import daft
        gray = ".3"
        pgm = daft.PGM((3.5, 4), node_ec=gray)
        scale = 1.5

        pgm.add_node(daft.Node("k", r"$k$", 1.5, 3.5, scale))
        pgm.add_node(daft.Node("vim1", r"$v_{i-1}$", .5, 2.5, scale))
        pgm.add_node(daft.Node("vi", r"$v_i$", 2.5, 2.5, scale))
        pgm.add_node(daft.Node("pim1", r"$p_{i-1}$", 1, 1.5, scale))
        pgm.add_node(daft.Node("pi", r"$p_i$", 3, 1.5, scale))
        pgm.add_node(daft.Node("yim1", r"$y_{i-1}$", 1, .5, scale, observed=True))
        pgm.add_node(daft.Node("yi", r"$y_i$", 3, .5, scale, observed=True))

        kws = {"plot_params": {"ec": gray, "fc": gray}}
        pgm.add_edge("k", "vim1", **kws)
        pgm.add_edge("k", "vi", **kws)
        pgm.add_edge("vim1", "pim1", **kws)
        pgm.add_edge("vi", "pi", **kws)
        pgm.add_edge("vim1", "vi", **kws)
        pgm.add_edge("pim1", "pi", **kws)
        pgm.add_edge("pim1", "yim1", **kws)
        pgm.add_edge("pi", "yi", **kws)

        pgm.render()
        return pgm


def make_grid(start, stop, step):
    """Define an even grid over a parameter space."""
    count = int((stop - start) / step + 1)
    return np.linspace(start, stop, count)


def I_trans_func(I_p1, I, k):
    """I_p1 is normal with mean I and std dev k."""
    var = exp(k * 2)
    pdf = exp(-.5 * power(I - I_p1, 2) / var)
    pdf *= power(2 * pi * var, -0.5)
    return pdf


def p_trans_func(p_p1, p, I_p1):
    """p_p1 is beta with mean p and precision I_p1."""
    a = 1 + exp(I_p1) * p
    b = 1 + exp(I_p1) * (1 - p)

    if 0 < p_p1 < 1:
        logkerna = (a - 1) * log(p_p1)
        logkernb = (b - 1) * log(1 - p_p1)
        betaln_ab = gammaln(a) + gammaln(b) - gammaln(a + b)
        return exp(logkerna + logkernb - betaln_ab)
    else:
        return 0
