import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
import time
from scipy.stats import norm
from collections import deque, defaultdict

# Disjoint Pattern Databases for 24-pancake
PDB_PATTERNS = [
    # First set of patterns (5-5-5-5-4)
    [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24]
    ],
    # Second set of patterns (4-5-5-5-5)
    [
        [1, 2, 3, 19],
        [4, 5, 6, 7, 8],
        [9, 10, 11, 12, 13],
        [14, 15, 16, 17, 18],
        [20, 21, 22, 23, 24]
    ]
]

class PancakePuzzle:
    def __init__(self, initial_state=None):
        self.goal_state = list(range(1, 25))
        self.initial_state = initial_state if initial_state else self.generate_random_state()
        
    def generate_random_state(self, steps=30):
        """Generate a random solvable pancake state"""
        state = self.goal_state.copy()
        for _ in range(steps):
            state = self.random_flip(state)
        return state
        
    def random_flip(self, state):
        """Perform a random flip on the state"""
        pos = random.randint(0, 22)  # Can't flip at last position
        return self.flip(state, pos)
        
    @staticmethod
    def flip(state, position):
        """Flip the stack at the given position (0-based index)"""
        return state[:position+1][::-1] + state[position+1:]
        
    def get_valid_moves(self, state):
        """Generate all possible next states by flipping at each possible position"""
        return [self.flip(state, i) for i in range(len(state)-1)]
        
    def is_goal(self, state):
        return state == self.goal_state
        
    def get_cost(self, state1, state2):
        return 1  # Uniform cost for all moves

def compute_pdb_heuristic(state, goal_state, pattern_group):
    """
    Compute PDB heuristic for pancake puzzle.
    For each pancake in the pattern, find its current position and goal position,
    then compute the minimum number of flips needed to move it to the correct position.
    """
    distance = 0
    goal_positions = {pancake: idx for idx, pancake in enumerate(goal_state)}
    
    for pancake in pattern_group:
        current_pos = state.index(pancake)
        goal_pos = goal_positions[pancake]
        
        # Each position difference requires at least one flip
        distance += abs(current_pos - goal_pos)
    
    return distance

def encode_pancake_state(state, goal_state):
    """Encode the pancake state into 14 features"""
    features = np.zeros(14)
    
    # Compute PDB features (f1-f10)
    for i, pattern_set in enumerate(PDB_PATTERNS):
        for j, pattern_group in enumerate(pattern_set):
            idx = i*5 + j
            if idx < 10:  # We have 10 pattern groups total (5+5)
                features[idx] = compute_pdb_heuristic(state, goal_state, pattern_group)
    
    # Sum of first set of PDBs (f11)
    features[10] = np.sum(features[:5])
    # Sum of second set of PDBs (f12)
    features[11] = np.sum(features[5:10])
    
    # Additional features
    # Binary feature indicating if middle pancake (position 12) is out of place
    middle_pos = 11  # 0-based index for position 12
    features[12] = 1 if state[middle_pos] != goal_state[middle_pos] else 0
    
    # Number of the largest out-of-place pancake
    largest_out_of_place = 0
    for i in range(1, 25):
        if state.index(i) != goal_state.index(i):
            largest_out_of_place = i
    features[13] = largest_out_of_place
    
    return features

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_mu=0.0, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.empty(out_features, in_features))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_logvar = nn.Parameter(torch.empty(out_features))
        
        # Initialize parameters
        nn.init.normal_(self.weight_mu, mean=prior_mu, std=prior_sigma/10)
        nn.init.constant_(self.weight_logvar, math.log(prior_sigma**2))
        nn.init.normal_(self.bias_mu, mean=prior_mu, std=prior_sigma/10)
        nn.init.constant_(self.bias_logvar, math.log(prior_sigma**2))
        
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

    def forward(self, x):
        weight_var = torch.exp(self.weight_logvar)
        bias_var = torch.exp(self.bias_logvar)
        
        mu_out = F.linear(x, self.weight_mu, self.bias_mu)
        var_out = F.linear(x.pow(2), weight_var, bias_var)
        
        eps = torch.randn_like(mu_out)
        return mu_out + torch.sqrt(var_out + 1e-8) * eps

    def kl_divergence(self):
        kl_weight = 0.5 * (
            (self.weight_mu - self.prior_mu).pow(2) + torch.exp(self.weight_logvar)
        ) / (self.prior_sigma**2) - 0.5 * (
            1 + self.weight_logvar - math.log(self.prior_sigma**2))
        
        kl_bias = 0.5 * (
            (self.bias_mu - self.prior_mu).pow(2) + torch.exp(self.bias_logvar)
        ) / (self.prior_sigma**2) - 0.5 * (
            1 + self.bias_logvar - math.log(self.prior_sigma**2))
            
        return kl_weight.sum() + kl_bias.sum()

class WUNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=8, S=5, prior_mu=0.0, prior_sigma=1.0):
        super().__init__()
        self.S = S
        self.fc1 = BayesianLinear(input_dim, hidden_dim, prior_mu, prior_sigma)
        self.fc2 = BayesianLinear(hidden_dim, 1, prior_mu, prior_sigma)

    def forward_single(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def predict_sigma_e(self, x, K=100):
        self.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            outputs = [self.forward_single(x_tensor).item() for _ in range(K)]
        return np.var(outputs)

class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # Outputs mean and variance
        
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        mean = output[:, 0]
        var = F.softplus(output[:, 1])  # Ensure variance is positive
        return mean, var

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            mean, var = self.forward(x)
            return mean.item(), var.item()

class PancakeSolver:
    def __init__(self, goal_state, params):
        self.puzzle = PancakePuzzle()
        self.goal_state = goal_state
        self.input_dim = 14  # Number of features
        
        # Initialize neural networks
        self.nnWUNN = WUNN(
            self.input_dim,
            params["hidden_dim"],
            prior_mu=params["mu0"],
            prior_sigma=math.sqrt(params["sigma2_0"]),
        )
        self.nnFFNN = FFNN(self.input_dim, params["hidden_dim"])
        
        # Learning parameters
        self.alpha = params["alpha0"]
        self.beta = params["beta0"]
        self.epsilon = params["epsilon"]
        self.delta = params["delta"]
        self.kappa = params["kappa"]
        self.gamma = (0.00001 / params["beta0"]) ** (1 / params["NumIter"])
        self.q = params["q"]
        self.K = params["K"]
        
        # Training parameters
        self.NumIter = params["NumIter"]
        self.NumTasksPerIter = params["NumTasksPerIter"]
        self.NumTasksPerIterThresh = params["NumTasksPerIterThresh"]
        self.TrainIter = params["TrainIter"]
        self.MaxTrainIter = params["MaxTrainIter"]
        self.MiniBatchSize = params["MiniBatchSize"]
        self.tmax = params["tmax"]
        
        # Memory buffer
        self.memoryBuffer = deque(maxlen=params["MemoryBufferMaxRecords"])
        
        # Tracking metrics
        self.planner_costs = []
        self.optimal_costs = []
        self.planning_times = []
        self.optimal_solutions_count = 0
        self.total_solved_tasks = 0

    def max_admissible_heuristic(self, state):
        """The maximum of the admissible PDB heuristics"""
        x = encode_pancake_state(state, self.goal_state)
        return np.max(x[:12])  # First 12 features are admissible

    def uncertainty_aware_heuristic(self, state):
        """Combine learned heuristic with admissible heuristic"""
        x = encode_pancake_state(state, self.goal_state)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        
        self.nnFFNN.eval()
        with torch.no_grad():
            mean, logvar = self.nnFFNN(x_tensor)
            y_hat = mean.item()
            sigma2_a = torch.exp(logvar).item()
        
        # Adjust uncertainty based on heuristic value
        sigma2_t = sigma2_a if y_hat < self.yq else self.epsilon
        
        # Compute uncertainty-aware heuristic value
        h_val = y_hat + math.sqrt(sigma2_t) * norm.ppf(self.alpha)
        
        # Get admissible heuristic value
        had = self.max_admissible_heuristic(state)
        
        return max(h_val, had)

    def ida_star(self, start_state, heuristic_func, tmax):
        """Iterative Deepening A* search with timeout"""
        threshold = heuristic_func(start_state)
        path = [start_state]
        total_nodes = 0
        start_time = time.time()
        
        def search(g, bound):
            nonlocal total_nodes
            if time.time() - start_time > tmax:
                raise TimeoutError()
            
            node = path[-1]
            f = g + heuristic_func(node)
            if f > bound:
                return f
            if self.puzzle.is_goal(node):
                return True
            
            min_t = float('inf')
            neighbors = self.puzzle.get_valid_moves(node)
            total_nodes += len(neighbors)
            
            for neighbor in neighbors:
                if neighbor in path:
                    continue
                path.append(neighbor)
                t = search(g + 1, bound)
                if t is True:
                    return True
                if t < min_t:
                    min_t = t
                path.pop()
            return min_t

        try:
            while True:
                t = search(0, threshold)
                if t is True:
                    planning_time = time.time() - start_time
                    return path, total_nodes, planning_time
                if t == float('inf'):
                    planning_time = time.time() - start_time
                    return None, total_nodes, planning_time
                threshold = t
        except TimeoutError:
            planning_time = time.time() - start_time
            return None, total_nodes, planning_time

    def train_ffnn(self):
        """Train the feedforward neural network"""
        if len(self.memoryBuffer) < self.MiniBatchSize:
            return

        optimizer = optim.Adam(self.nnFFNN.parameters())
        criterion = nn.GaussianNLLLoss()
        
        # Prepare data
        x_data = torch.stack(
            [torch.tensor(x, dtype=torch.float32) for x, _ in self.memoryBuffer]
        )
        y_data = torch.tensor(
            [y for _, y in self.memoryBuffer], dtype=torch.float32
        ).unsqueeze(1)

        self.nnFFNN.train()
        for _ in range(self.TrainIter):
            permutation = torch.randperm(len(x_data))
            for i in range(0, len(x_data), self.MiniBatchSize):
                indices = permutation[i : i + self.MiniBatchSize]
                x_batch, y_batch = x_data[indices], y_data[indices]
                
                optimizer.zero_grad()
                mean, var = self.nnFFNN(x_batch)
                loss = criterion(mean, y_batch, var)
                loss.backward()
                optimizer.step()

    def train_wunn(self):
        """Train the Weighted Uncertainty Neural Network"""
        if len(self.memoryBuffer) < self.MiniBatchSize:
            return False

        self.nnWUNN.train()
        optimizer = optim.Adam(self.nnWUNN.parameters(), lr=0.01)
        early_stop = False
        
        # Calculate uncertainties for all samples
        uncertainties = []
        for x, _ in self.memoryBuffer:
            sigma2_e = max(self.nnWUNN.predict_sigma_e(x, K=10), 1e-8)
            uncertainties.append(sigma2_e)
        
        # Calculate weights with overflow protection
        weights = []
        for sigma2_e in uncertainties:
            try:
                if sigma2_e >= self.kappa * self.epsilon:
                    exponent = min(math.sqrt(sigma2_e), 50)
                    weight = math.exp(exponent)
                else:
                    weight = math.exp(-1)
                weights.append(weight)
            except OverflowError:
                weights.append(1e20)
        
        # Normalize weights
        max_weight = max(weights)
        if max_weight > 1e6:
            weights = [w/max_weight*1e6 for w in weights]
        
        # Training loop
        for iter in range(self.MaxTrainIter):
            # Early stopping check
            if iter % 10 == 0:
                all_low_uncertainty = True
                for x, _ in list(self.memoryBuffer)[:100]:
                    sigma2_e = self.nnWUNN.predict_sigma_e(x, 10)
                    if sigma2_e >= self.kappa * self.epsilon:
                        all_low_uncertainty = False
                        break
                if all_low_uncertainty:
                    early_stop = True
                    break

            # Sample weighted mini-batch
            batch_indices = random.choices(
                range(len(self.memoryBuffer)),
                weights=weights,
                k=min(self.MiniBatchSize, len(self.memoryBuffer))
            )
            batch = [self.memoryBuffer[i] for i in batch_indices]

            # Training step
            total_loss = 0
            for x, y in batch:
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                y_tensor = torch.tensor([y], dtype=torch.float32).unsqueeze(1)
                
                # Forward pass with multiple samples
                preds = torch.stack(
                    [self.nnWUNN.forward_single(x_tensor) for _ in range(self.nnWUNN.S)]
                )
                pred_mean = preds.mean(dim=0)
                
                # Loss calculation
                log_likelihood = -F.mse_loss(pred_mean, y_tensor)
                kl_div = self.nnWUNN.fc1.kl_divergence() + self.nnWUNN.fc2.kl_divergence()
                loss = self.beta * kl_div - log_likelihood
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        return early_stop

    def run(self):
        """Main training and solving loop"""
        self.yq = -np.inf  # Initialize yq to negative infinity
        
        # Tracking metrics
        all_alphas = []
        all_suboptimalities = []
        all_optimal_rates = []
        all_planning_times = []
        all_nodes_generated = []
        
        print("Iter\tα\tTime\tSubopt%\tOpt%\tGenerated")

        for n in range(1, self.NumIter + 1):
            iter_planner_costs = []
            iter_optimal_costs = []
            iter_solved_count = 0
            iter_times = []
            iter_nodes = []
            
            for _ in range(self.NumTasksPerIter):
                # Generate a random task
                task = self.puzzle.generate_random_state()
                
                try:
                    start_time = time.time()
                    # Solve using IDA* with uncertainty-aware heuristic
                    plan, nodes_generated, planning_time = self.ida_star(
                        task,
                        self.uncertainty_aware_heuristic,
                        self.tmax
                    )
                    
                    iter_times.append(planning_time)
                    iter_nodes.append(nodes_generated)
                    
                    if plan:
                        plan_cost = len(plan) - 1
                        optimal_cost = self.max_admissible_heuristic(task)
                        
                        iter_planner_costs.append(plan_cost)
                        iter_optimal_costs.append(optimal_cost)
                        iter_solved_count += 1
                        
                        self.planner_costs.append(plan_cost)
                        self.optimal_costs.append(optimal_cost)
                        self.total_solved_tasks += 1
                        
                        # Consider solution optimal if within 5% of estimated optimal
                        if plan_cost <= optimal_cost * 1.05:
                            self.optimal_solutions_count += 1
                        
                        # Store the solution path in memory buffer
                        for state in reversed(plan[:-1]):
                            x = encode_pancake_state(state, self.goal_state)
                            y = self.max_admissible_heuristic(state)
                            self.memoryBuffer.appendleft((x, y))
                            
                except TimeoutError:
                    planning_time = time.time() - start_time
                    iter_times.append(planning_time)
                    iter_nodes.append(self.tmax * 1000)  # Large number for timeout

            # Adjust alpha based on performance
            if iter_solved_count < self.NumTasksPerIterThresh:
                self.alpha = max(self.alpha - self.delta, 0.01)
            else:
                self.alpha = min(self.alpha + self.delta/2, 0.99)

            # Train the networks
            self.train_ffnn()
            early_stop = self.train_wunn()
            
            # Adjust beta for KL divergence
            self.beta *= self.gamma
            
            # Calculate iteration metrics
            if iter_solved_count > 0:
                subopt_pct = [((y/y_star)-1)*100 for y, y_star in zip(iter_planner_costs, iter_optimal_costs)]
                avg_subopt_pct = sum(subopt_pct)/len(subopt_pct)
                opt_count = sum(1 for y, y_star in zip(iter_planner_costs, iter_optimal_costs) 
                              if y <= y_star * 1.05)
                opt_rate = (opt_count/iter_solved_count)*100
            else:
                avg_subopt_pct = 0
                opt_rate = 0
                
            avg_time_iter = np.mean(iter_times) if iter_times else 0
            avg_nodes_iter = np.mean(iter_nodes) if iter_nodes else 0
            
            # Store metrics
            all_alphas.append(self.alpha)
            if iter_solved_count > 0:
                all_suboptimalities.append(avg_subopt_pct)
                all_optimal_rates.append(opt_rate)
            all_planning_times.append(avg_time_iter)
            all_nodes_generated.append(avg_nodes_iter)
            
            print(f"{n}\t{self.alpha:.3f}\t{avg_time_iter:.2f}\t{avg_subopt_pct:.1f}%\t{opt_rate:.1f}%\t{avg_nodes_iter:.0f}")

            # Early stopping if uncertainty is low
            if early_stop and n > 10:
                print("Early stopping due to low uncertainty")
                break

        # Calculate final metrics
        final_avg_alpha = np.mean(all_alphas)
        final_avg_time = np.mean(all_planning_times)
        final_avg_nodes = np.mean(all_nodes_generated)
        
        valid_subopts = [s for s in all_suboptimalities if s != 0]
        final_avg_subopt = np.mean(valid_subopts) if valid_subopts else 0
        
        valid_opts = [o for o in all_optimal_rates if o != 0]
        final_avg_opt = np.mean(valid_opts) if valid_opts else 0

        # Global metrics
        if self.total_solved_tasks > 0:
            global_subopt_pct = [((y/y_star)-1)*100 for y, y_star in zip(self.planner_costs, self.optimal_costs)]
            global_avg_subopt = sum(global_subopt_pct)/len(global_subopt_pct)
            global_opt_rate = (self.optimal_solutions_count/self.total_solved_tasks)*100
        else:
            global_avg_subopt = 0
            global_opt_rate = 0

        print("\n=== Final Results ===")
        print(f"Total tasks attempted: {self.NumIter * self.NumTasksPerIter}")
        print(f"Total tasks solved: {self.total_solved_tasks}")
        print(f"Optimal solutions found: {self.optimal_solutions_count} ({global_opt_rate:.1f}%)")
        print(f"Average suboptimality: {global_avg_subopt:.1f}%")
        print(f"Average planning time: {final_avg_time:.2f} seconds")
        print(f"Average nodes generated: {final_avg_nodes:.0f}")
        
        print("\n=== Final Averages ===")
        print("Iter\tα\tTime\tSubopt%\tOpt%\tGenerated")
        print(f"All\t{final_avg_alpha:.3f}\t{final_avg_time:.2f}\t{final_avg_subopt:.1f}%\t{final_avg_opt:.1f}%\t{final_avg_nodes:.0f}")
        
        return {
            "alpha_history": all_alphas,
            "suboptimalities": all_suboptimalities,
            "optimal_rates": all_optimal_rates,
            "planning_times": all_planning_times,
            "nodes_generated": all_nodes_generated,
            "final_metrics": {
                "alpha": final_avg_alpha,
                "suboptimality": global_avg_subopt,
                "optimal_rate": global_opt_rate,
                "planning_time": final_avg_time,
                "nodes_generated": final_avg_nodes
            }
        }

if __name__ == "__main__":
    # Parameters for the solver
    params = {
        "hidden_dim": 8,
        "alpha0": 0.99,
        "beta0": 0.05,
        "epsilon": 1.0,
        "delta": 0.05,
        "kappa": 0.64,
        "q": 0.95,
        "K": 10,
        "mu0": 0.0,
        "sigma2_0": 10.0,
        "NumIter": 50,
        "NumTasksPerIter": 5,
        "NumTasksPerIterThresh": 3,
        "TrainIter": 100,
        "MaxTrainIter": 500,
        "MiniBatchSize": 32,
        "tmax": 10,  # 5 minutes timeout
        "MemoryBufferMaxRecords": 5000,
    }

    # Initialize and run the solver
    goal_state = list(range(1, 25))  # [1, 2, ..., 24]
    solver = PancakeSolver(goal_state, params)
    results = solver.run()