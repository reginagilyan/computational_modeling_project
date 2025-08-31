

### import block

from mesa import Agent
import numpy as np
from scipy.stats import beta
from scipy.optimize import minimize_scalar
from fractions import Fraction
from mesa import Model
from mesa.space import SingleGrid
from mesa.time import SimultaneousActivation
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pickle


# set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

#########################################################
# AGENT CLASS
#########################################################

class SSTAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        # sample private mean attitude from Beta(10, 10)
        self.private_mean = round(self.random.betavariate(10, 10), 2)

        # convert to smallest integers α and β using rational approximation
        frac = Fraction(self.private_mean).limit_denominator(100)
        self.alpha_private = frac.numerator
        self.beta_private = frac.denominator - self.alpha_private

        # initial expressed attitude = private mean
        self.A_i = self.private_mean

        # parameters 
        self.w = model.w
        self.gamma = model.gamma

    def utility(self, A_i, alpha_social, beta_social):
        """
        calculates utility as the negative of a weighted sum of social and private disutilities 
        based on the expressed attitude and beta distribution parameters

        Args:
            A_i (float): expressed attitude value to evaluate
            alpha_social (int): alpha parameter of the beta distribution representing social attitudes.
            beta_social (int): beta parameter of the beta distribution representing social attitudes.

        Returns:
            float: calculated utility (negative disutility) value
        """
        
        # CDF of expressed attitude with respect to social and private attitude distributions
        I_social = beta.cdf(A_i, alpha_social, beta_social)
        I_private = beta.cdf(A_i, self.alpha_private, self.beta_private)

        # disutilities based on difference from center (0.5)
        social_disutility = np.exp(self.gamma * (abs(I_social - 0.5) - 0.5))
        private_disutility = np.exp(self.gamma * (abs(I_private - 0.5) - 0.5))

        # combine disutilities, weighted by w, and return negative disutility
        total_disutility = self.w * social_disutility + (1 - self.w) * private_disutility
        return -total_disutility  # return negative disutility as utility

    
    def fit_beta(self, values, max_sum=20):
        """
        finds the integer alpha and beta parameters of a Beta distribution that best fit the given values
        using maximum log-likelihood (LL) estimation; subject to the constraint that alpha + beta <= max_sum

        Args:
            values (array-like): sample data to fit the Beta distribution to
            max_sum (int, optional): maximum allowed sum of alpha and beta parameters (20 as per SST paper)

        Returns:
            tuple: best-fitting (alpha, beta) parameters as integers
        """
        values = np.array(values)
        best_alpha, best_beta = 1, 1
        best_log_likelihood = -np.inf

        # iterate over all valid alpha values
        for alpha in range(1, max_sum):
            # define possible beta values (such that alpha + beta <= max_sum)
            beta_range = np.arange(1, max_sum - alpha + 1)
            alphas = np.full_like(beta_range, alpha)

            # compute LL for all alpha-beta-pairs
            logpdfs = beta.logpdf(values[:, None], alphas, beta_range)  # shape: (n_values, n_beta)
            log_likelihoods = logpdfs.sum(axis=0)  # sum LL for each beta

            # find the best beta (maximum LL) for current alpha
            max_idx = np.argmax(log_likelihoods)
            if log_likelihoods[max_idx] > best_log_likelihood:
                best_log_likelihood = log_likelihoods[max_idx]
                best_alpha = alpha
                best_beta = beta_range[max_idx]

        return best_alpha, best_beta

    
    def swap_utility(self, neighbors):
        """
        calculates utility of the agent (=A_i, = expressed attitude) given the attitudes of its neighbors

        Args:
            neighbors (list): list of neighbor agents whose A_i values are used

        Returns:
            float: utility value based on fitted beta distribution
        """
        alpha, beta_ = self.fit_beta([a.A_i for a in neighbors])
        return self.utility(self.A_i, alpha, beta_)
    
    def step(self):
        """

        steps:
        1. retrieve expressed attitudes of the agent's neighbors
        2. fit a beta distribution to neighbors' attitudes to estimate social norm
        3. find the attitude value in [0, 1] that maximizes the agent's utility given the estimated social norm
        4. update agent's expressed attitude (A_i) to this utility-maximizing value

        """

        # step 1
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        neighbor_attitudes = [agent.A_i for agent in neighbors if hasattr(agent, "A_i")]

        # step 2
        alpha_est, beta_est = self.fit_beta(neighbor_attitudes, max_sum=20)

        # step 3
        result = minimize_scalar(
            lambda x: -self.utility(x, alpha_est, beta_est),  # minimize negative disutility = maximize utility
            bounds=(0, 1),
            method='bounded'
        )

        # step 4
        self.A_i = result.x

#########################################################
# MODEL CLASS
#########################################################

class SSTModel(Model):
    def __init__(self, width=100, height=100, seed=SEED, n_swaps=1, max_attempts=200, w=0.5, gamma=20, max_cycles=500):
        super().__init__(seed=seed)
        """
        Args:
            width (int): grid width, default = 100
            height (int): grid height, default = 100
            seed (int): random seed for reproducibility
            n_swaps (int): number of utility-improving swaps attempted per cycle
            max_attempts (int): maximum attempts to find a swap per cycle
            w (int): social comparison parameter 
            gamma (int): 
        """
        self.width = width
        self.height = height

        self.num_agents = width * height
        self.seed = seed

        self.w = w
        self.gamma = gamma

        self.run_in_period = 2      # no movement for first 2 cycles
        self.current_cycle = -2     # cycle number 1 will be the first cycle where a swap happens
        self.n_swaps = n_swaps
        self.max_attempts = max_attempts

        # track no swap streak for early stopping
        self.no_swap_streak = 0
        self.early_stop_threshold = 30
        self.running = True

        # track and store cycles where no swaps happen
        self.swapless_cycles = []
        self.swapless_count = 0

        self.grid = SingleGrid(width, height, torus=True)
        self.agent_list = []        # creating agent list instead of using scheduler to improve parallelization

        # create agents
        agent_id = 0
        for cell_content, (x, y) in self.grid.coord_iter():
            agent = SSTAgent(agent_id, self)
            self.agent_list.append(agent) 
            self.grid.place_agent(agent, (x, y))
            agent_id += 1

        # for storing metrics
        self.expressed_over_time = {}
        self.variance_over_time = {}
        self.expressed_snapshots = {}
        self.disutility_over_time = {}

        self.snapshot_cycles = [-2, 0]
        self.max_cycles = max_cycles


    def _estimate_social_norm(self, agent):
        """
        helper function to estimate social norm for the purpose of storing this metric

        Args:
            agent: agent for whom to estimate the social norm

        Returns:
            tuple: alpha-beta-tuple representing the fitted beta distribution parameters
        """
        neighbors = self.grid.get_neighbors(agent.pos, moore=True, include_center=False)
        neighbor_attitudes = [a.A_i for a in neighbors if hasattr(a, "A_i")]
        return agent.fit_beta(neighbor_attitudes)


    def step(self):
        """
        steps:
        1. all agents update their expressed attitudes in parallel
        2. in initial "run-in" period (cycles < 0), no swaps are performed
        3. after stabilization, attempts to find utility-improving agent swaps
        (attempts up to `max_attempts`, number of swaps per cycle defined by 'n_swaps')

        """
        self.current_cycle += 1

        # step 1
        with ThreadPoolExecutor() as executor:
            list(executor.map(lambda a: a.step(), self.agent_list))

        # step 2
        if self.current_cycle < 0:
            return 

        # step 3
        successful_swaps = 0
        attempts = 0

        while successful_swaps < self.n_swaps and attempts < self.max_attempts:
            attempts += 1

            # sample two random agents
            agent1, agent2 = self.random.sample(self.agent_list, 2)
            # record position of agents
            pos1 = agent1.pos
            pos2 = agent2.pos

            # get neighbors in both positions
            neighbors1 = [a for a in self.grid.get_neighbors(pos1, moore=True, include_center=False) if a != agent1]
            neighbors2 = [a for a in self.grid.get_neighbors(pos2, moore=True, include_center=False) if a != agent2]

            # calculate utilities before and after potential swap
            u1_current = agent1.swap_utility(neighbors1)
            u1_swapped = agent1.swap_utility(neighbors2)
            u2_current = agent2.swap_utility(neighbors2)
            u2_swapped = agent2.swap_utility(neighbors1)

            # swap using helper function IF both agents would have increased utility on other position
            if u1_swapped > u1_current and u2_swapped > u2_current:
                self.grid.swap_pos(agent1, agent2)         
                successful_swaps += 1

        # print statement if no swap happened
        if successful_swaps == 0:
            print(f"No successful swap on cycle {self.current_cycle}.")
            self.no_swap_streak += 1
            self.swapless_count += 1
            self.swapless_cycles.append(self.current_cycle)
        else:
            self.no_swap_streak = 0

        # check early stopping condition
        if self.no_swap_streak >= self.early_stop_threshold:
            print(f"Early stopping at cycle {self.current_cycle} — no swaps for {self.no_swap_streak} consecutive cycles.")
            self.running = False

        # collect data
        if self.current_cycle % 50 == 0:
            expressed = [agent.A_i for agent in self.agent_list]
            self.expressed_over_time[self.current_cycle] = expressed
            self.variance_over_time[self.current_cycle] = np.var(expressed)
            total_disutility = sum(
                -agent.utility(agent.A_i, *self._estimate_social_norm(agent))
                for agent in self.agent_list
            )
            self.disutility_over_time[self.current_cycle] = total_disutility

        # dynamic snapshot collection
        expressed = [agent.A_i for agent in self.agent_list]
        if self.current_cycle in self.snapshot_cycles:
            self.expressed_snapshots[self.current_cycle] = expressed
        elif self.max_cycles > 0:
            checkpoints = [
                int(self.max_cycles * 0.25),
                int(self.max_cycles * 0.50),
                int(self.max_cycles * 0.75),
            ]
            if self.current_cycle in checkpoints:
                self.expressed_snapshots[self.current_cycle] = expressed

        # always store the final state
        if not self.running:
            self.expressed_snapshots[self.current_cycle] = expressed



#########################################################
# RUN MODEL
#########################################################

# initialize
model = SSTModel(width=30, height=30, max_cycles=800)

# run
with tqdm(total=model.max_cycles, desc="Running SST Model") as pbar:
    while model.running and model.current_cycle < model.max_cycles:
        model.step()
        pbar.update(1)


# extract final snapshot
if model.current_cycle not in model.expressed_snapshots:
    model.expressed_snapshots[model.current_cycle] = [
        agent.A_i for agent in model.agent_list
    ]
    print(f"Final snapshot manually saved at cycle {model.current_cycle}")


#########################################################
# EXPORT
#########################################################

# save data
model_output = {
    "expressed_over_time": model.expressed_over_time,
    "variance_over_time": model.variance_over_time,
    "disutility_over_time": model.disutility_over_time,
    "expressed_snapshots": model.expressed_snapshots,
    "swapless_cycles": model.swapless_cycles,
    "snapshot_cycles": model.snapshot_cycles,
    "final_cycle": model.current_cycle
}

with open("output.pkl", "wb") as f:
    pickle.dump(model_output, f)

print("Model outputs saved as 'output.pkl'")
