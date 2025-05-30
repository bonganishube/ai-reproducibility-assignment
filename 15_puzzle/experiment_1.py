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


# --- Environment and Utility Functions ---
def encode_15puzzle_state(state):
    """Encodes the 15-puzzle state into a feature vector"""
    encoded = np.zeros(16 * 2 * 4)
    for tile in range(16):
        idx = state.index(tile)
        row, col = divmod(idx, 4)
        encoded[tile * 8 + row] = 1
        encoded[tile * 8 + 4 + col] = 1
    return encoded


def get_valid_moves(state):
    """Returns all valid moves from the current state"""
    zero_index = state.index(0)
    row, col = divmod(zero_index, 4)
    valid_moves = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < 4 and 0 <= new_col < 4:
            new_zero_index = new_row * 4 + new_col
            new_state = state[:]
            new_state[zero_index], new_state[new_zero_index] = (
                new_state[new_zero_index],
                new_state[zero_index],
            )
            valid_moves.append(new_state)
    return valid_moves


def manhattan_distance(state, goal_state):
    """Computes Manhattan distance heuristic"""
    total = 0
    for i in range(1, 16):
        curr_idx = state.index(i)
        goal_idx = goal_state.index(i)
        curr_row, curr_col = divmod(curr_idx, 4)
        goal_row, goal_col = divmod(goal_idx, 4)
        total += abs(curr_row - goal_row) + abs(curr_col - goal_col)
    return total


# --- Neural Network Definitions ---
class BayesianLinear(nn.Module):
    """Bayesian linear layer with local reparameterization"""

    def __init__(self, in_features, out_features, prior_mu=0.0, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize parameters with given priors
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_logvar = nn.Parameter(torch.empty(out_features))

        # Initialize with prior values
        nn.init.normal_(self.weight_mu, mean=prior_mu, std=prior_sigma / 10)
        nn.init.constant_(self.weight_logvar, math.log(prior_sigma**2))
        nn.init.normal_(self.bias_mu, mean=prior_mu, std=prior_sigma / 10)
        nn.init.constant_(self.bias_logvar, math.log(prior_sigma**2))

        # Prior parameters
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

    def forward(self, x):
        # Local reparameterization trick
        weight_var = torch.exp(self.weight_logvar)
        bias_var = torch.exp(self.bias_logvar)

        mu_out = F.linear(x, self.weight_mu, self.bias_mu)
        var_out = F.linear(x.pow(2), weight_var, bias_var)

        eps = torch.randn_like(mu_out)
        return mu_out + torch.sqrt(var_out + 1e-8) * eps

    def kl_divergence(self):
        """Computes KL divergence between posterior and prior"""
        kl_weight = 0.5 * (
            (self.weight_mu - self.prior_mu).pow(2) + torch.exp(self.weight_logvar)
        ) / (self.prior_sigma**2) - 0.5 * (
            1 + self.weight_logvar - math.log(self.prior_sigma**2)
        )
        kl_bias = 0.5 * (
            (self.bias_mu - self.prior_mu).pow(2) + torch.exp(self.bias_logvar)
        ) / (self.prior_sigma**2) - 0.5 * (
            1 + self.bias_logvar - math.log(self.prior_sigma**2)
        )
        return kl_weight.sum() + kl_bias.sum()


class WUNN(nn.Module):
    """Weight Uncertainty Neural Network for epistemic uncertainty"""

    def __init__(self, input_dim, hidden_dim=20, S=5, prior_mu=0.0, prior_sigma=1.0):
        super().__init__()
        self.S = S  # Number of samples for forward pass
        self.fc1 = BayesianLinear(input_dim, hidden_dim, prior_mu, prior_sigma)
        self.fc2 = BayesianLinear(hidden_dim, 1, prior_mu, prior_sigma)

    def forward_single(self, x):
        """Single forward pass"""
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def predict_sigma_e(self, x, K=100):
        """Predicts epistemic uncertainty"""
        self.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            outputs = [self.forward_single(x_tensor).item() for _ in range(K)]
        return np.var(outputs)


class FFNN(nn.Module):
    """Feedforward Neural Network for aleatoric uncertainty"""

    def __init__(self, input_dim, hidden_dim=20, dropout_rate=0.025):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, 1)
        self.fc2_var = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2_mean.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2_var.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2_mean.bias)
        nn.init.zeros_(self.fc2_var.bias)

    def forward(self, x):
        """Forward pass returning mean and log variance"""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2_mean(x), self.fc2_var(x)

    def predict(self, x):
        """Predicts mean and variance"""
        self.eval()
        with torch.no_grad():
            mean, logvar = self.forward(x)
            return mean.item(), torch.exp(logvar).item()


# --- Main Algorithm Implementation ---
class LearnHeuristicPrac:
    def __init__(self, input_dim, goal_state, params):
        # Initialize models with prior parameters
        self.nnWUNN = WUNN(
            input_dim,
            params["hidden_dim"],
            prior_mu=params["mu0"],
            prior_sigma=math.sqrt(params["sigma2_0"]),
        )
        self.nnFFNN = FFNN(input_dim, params["hidden_dim"], params["dropout_rate"])        

        # Algorithm parameters
        self.alpha = params["alpha0"]
        self.beta = params["beta0"]
        self.epsilon = params["epsilon"]
        self.delta = params["delta"]
        self.kappa = params["kappa"]
        self.gamma = (0.00001 / params["beta0"]) ** (1 / params["NumIter"])  # Key change: Compute γ
        self.q = params["q"]
        self.K = params["K"]
        self.max_steps = params["MaxSteps"]

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

        # Metrics tracking
        self.planner_costs = []
        self.optimal_costs = []
        self.planning_times = []  # To store planning times for each task
        self.suboptimalities = []
        self.optimality_counts = 0
        self.goal_state = goal_state

    def h(self, alpha, mu, sigma):
        """Quantile function of normal distribution"""
        return mu + sigma * norm.ppf(alpha)

    def generate_task(self):
        """Generates a task with high epistemic uncertainty (Algorithm 3)"""
        s_prime = self.goal_state[:]
        s_double_prime = None

        for _ in range(self.max_steps):
            states = {}
            valid_moves = get_valid_moves(s_prime)

            for s in valid_moves:
                if s_double_prime is not None and s == s_double_prime:
                    continue

                x = encode_15puzzle_state(s)
                sigma2_e = self.nnWUNN.predict_sigma_e(x, self.K)
                states[tuple(s)] = sigma2_e  # Use tuple as dict key

            if not states:
                break

            # Softmax sampling
            states_list = list(states.items())
            state_tuples, sigmas = zip(*states_list)
            probs = F.softmax(torch.tensor(sigmas), dim=0).numpy()
            selected_idx = np.random.choice(len(state_tuples), p=probs)
            selected_state = list(state_tuples[selected_idx])
            selected_sigma = sigmas[selected_idx]

            if selected_sigma >= self.epsilon:
                return {
                    "s": selected_state,
                    "sg": self.goal_state,
                    "sigma2_e": selected_sigma,
                }

            s_double_prime = s_prime
            s_prime = selected_state

        return None

    def ida_star(self, start, goal, heuristic, tmax, start_time):
        """IDA* implementation with node counting and time tracking"""
        threshold = heuristic(start)
        path = [start]
        total_nodes = 0
        
        def search(g, bound):
            nonlocal total_nodes
            if time.time() - planning_start > tmax:  # Use planning_start instead of start_time
                raise TimeoutError()
            
            node = path[-1]
            f = g + heuristic(node)
            if f > bound:
                return f
            if node == goal:
                return True
            
            min_t = float('inf')
            neighbors = get_valid_moves(node)
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

        planning_start = time.time()  # Start timing here, right before the main loop
        try:
            while True:
                t = search(0, threshold)
                if t is True:
                    planning_time = time.time() - planning_start
                    return path, total_nodes, planning_time
                if t == float('inf'):
                    planning_time = time.time() - planning_start
                    return None, total_nodes, planning_time
                threshold = t
        except TimeoutError:
            planning_time = time.time() - planning_start
            return None, total_nodes, planning_time

    def uncertainty_aware_heuristic(self, state):
        """Computes h(s) = max(h(α, ŷ(s), σ_t(s)), 0)"""
        x = encode_15puzzle_state(state)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        # Get FFNN predictions
        self.nnFFNN.eval()
        with torch.no_grad():
            mean, logvar = self.nnFFNN(x_tensor)
            y_hat = mean.item()
            sigma2_a = torch.exp(logvar).item()

        # Determine which variance to use
        sigma2_t = sigma2_a if y_hat < self.yq else self.epsilon

        # Compute heuristic value
        h_val = self.h(self.alpha, y_hat, math.sqrt(sigma2_t))
        return max(h_val, 0)

    def compute_metrics(self):
        """Compute suboptimality, optimality, and timing metrics"""
        if not self.planner_costs:
            return 0.0, 0.0, 0.0

        # Existing suboptimality calculation
        suboptimalities = [
            (y / y_star) - 1
            for y, y_star in zip(self.planner_costs, self.optimal_costs)
            if y_star > 0
        ]
        avg_suboptimality = sum(suboptimalities) / len(suboptimalities) if suboptimalities else 0.0

        # Existing optimality rate calculation
        optimality_rate = (self.optimality_counts / len(self.planner_costs)) * 100 if self.planner_costs else 0.0

        # New timing metrics
        avg_planning_time = sum(self.planning_times) / len(self.planning_times) if self.planning_times else 0.0

        return avg_suboptimality, optimality_rate, avg_planning_time

    def train_ffnn(self):
        """Trains FFNN on entire memory buffer"""
        if len(self.memoryBuffer) < self.MiniBatchSize:
            return

        optimizer = optim.Adam(self.nnFFNN.parameters())
        criterion = nn.GaussianNLLLoss()

        # Convert memory buffer to tensors
        x_data = torch.stack(
            [torch.tensor(x, dtype=torch.float32) for x, _ in self.memoryBuffer]
        )
        y_data = torch.tensor(
            [y for _, y in self.memoryBuffer], dtype=torch.float32
        ).unsqueeze(1)

        self.nnFFNN.train()
        for _ in range(self.TrainIter):
            # Shuffle and batch the data
            permutation = torch.randperm(len(x_data))
            for i in range(0, len(x_data), self.MiniBatchSize):
                indices = permutation[i : i + self.MiniBatchSize]
                x_batch, y_batch = x_data[indices], y_data[indices]

                optimizer.zero_grad()
                mean, logvar = self.nnFFNN(x_batch)
                loss = criterion(mean, y_batch, torch.exp(logvar))
                loss.backward()
                optimizer.step()

    def train_wunn(self):
        """Trains WUNN with prioritized sampling and early stopping."""
        if len(self.memoryBuffer) < self.MiniBatchSize:
            return False

        self.nnWUNN.train()
        optimizer = optim.Adam(self.nnWUNN.parameters(), lr=0.01)
        early_stop = False

        # Precompute epistemic uncertainties for the entire buffer
        uncertainties = []
        for x, _ in self.memoryBuffer:
            sigma2_e = self.nnWUNN.predict_sigma_e(x, K=10)  # Approximate σ²_e
            uncertainties.append(sigma2_e)

        # Compute sampling weights
        weights = []
        for sigma2_e in uncertainties:
            if sigma2_e >= self.kappa * self.epsilon:
                weight = math.exp(math.sqrt(sigma2_e))  # exp(σ_e)
            else:
                weight = math.exp(-1)  # C=1
            weights.append(weight)

        for iter in range(self.MaxTrainIter):
            # Early stopping check (unchanged)
            if iter % 10 == 0:
                all_low_uncertainty = True
                for x, _ in list(self.memoryBuffer)[:100]:  # Check subset
                    sigma2_e = self.nnWUNN.predict_sigma_e(x, 10)
                    if sigma2_e >= self.kappa * self.epsilon:
                        all_low_uncertainty = False
                        break
                if all_low_uncertainty:
                    early_stop = True
                    break

            # --- Prioritized Sampling ---
            # Sample indices based on weights
            batch_indices = random.choices(
                range(len(self.memoryBuffer)),
                weights=weights,
                k=min(self.MiniBatchSize, len(self.memoryBuffer))
            )
            batch = [self.memoryBuffer[i] for i in batch_indices]

            # Training loop (fixed)
            total_loss = 0
            for x, y in batch:
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                y_tensor = torch.tensor([y], dtype=torch.float32).unsqueeze(1)  # Fixed: Ensure shape [1,1]
                
                preds = torch.stack(
                    [self.nnWUNN.forward_single(x_tensor) for _ in range(self.nnWUNN.S)]
                )
                pred_mean = preds.mean(dim=0)  # Shape [1,1]
                
                # Ensure shapes match: [1,1] vs [1,1]
                log_likelihood = -F.mse_loss(pred_mean, y_tensor)
                kl_div = self.nnWUNN.fc1.kl_divergence() + self.nnWUNN.fc2.kl_divergence()
                loss = self.beta * kl_div - log_likelihood
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        return early_stop

    def run(self):
        """Main learning loop with node generation tracking, planning time, and strict β decay"""
        self.yq = -np.inf
        # Initialize statistics tracking
        node_stats = {
            'total_nodes': [],
            'tasks_solved': 0,
            'nodes_per_solved': [],
            'nodes_per_unsolved': [],
            'planning_times': [],  # New: track all planning times
            'solved_times': [],    # New: track times for solved tasks
            'unsolved_times': []   # New: track times for unsolved tasks
        }

        for n in range(1, self.NumIter + 1):
            print(f"\n=== Iteration {n}/{self.NumIter} ===")
            print(f"Current β: {self.beta:.6f}")

            # Update yq from memory buffer
            if self.memoryBuffer:
                costs = [y for _, y in self.memoryBuffer]
                self.yq = np.quantile(costs, self.q)
            print(f"Current yq (q={self.q}): {self.yq:.2f}, α: {self.alpha:.3f}")

            # Generate and solve tasks
            numSolved = 0
            iter_nodes = []  # Track nodes per iteration
            iter_times = []  # Track planning times per iteration
            
            for i in range(self.NumTasksPerIter):
                T = self.generate_task()
                if not T:
                    print("No task generated (low uncertainty)")
                    continue

                try:
                    start_time = time.time()
                    plan, nodes_generated, planning_time = self.ida_star(  # Now returns planning_time
                        T["s"], T["sg"], 
                        self.uncertainty_aware_heuristic,
                        self.tmax, start_time
                    )
                    
                    # Record statistics
                    node_stats['total_nodes'].append(nodes_generated)
                    node_stats['planning_times'].append(planning_time)
                    iter_nodes.append(nodes_generated)
                    iter_times.append(planning_time)
                    
                    if plan:
                        numSolved += 1
                        node_stats['tasks_solved'] += 1
                        node_stats['nodes_per_solved'].append(nodes_generated)
                        node_stats['solved_times'].append(planning_time)
                        
                        plan_cost = len(plan) - 1
                        optimal_cost = manhattan_distance(T["s"], T["sg"])
                        self.planner_costs.append(plan_cost)
                        self.optimal_costs.append(optimal_cost)
                        
                        if plan_cost == optimal_cost:
                            self.optimality_counts += 1
                            
                        print(f"✓ Task {i+1}: cost={plan_cost}, optimal={optimal_cost}, "
                            f"nodes={nodes_generated}, time={planning_time:.2f}s")
                        
                        for state in reversed(plan[:-1]):
                            x = encode_15puzzle_state(state)
                            y = manhattan_distance(state, T["sg"])
                            self.memoryBuffer.appendleft((x, y))
                    else:
                        node_stats['nodes_per_unsolved'].append(nodes_generated)
                        node_stats['unsolved_times'].append(planning_time)
                        print(f"✗ Task {i+1}: failed, nodes={nodes_generated}, time={planning_time:.2f}s")
                        
                except TimeoutError:
                    planning_time = time.time() - start_time
                    node_stats['nodes_per_unsolved'].append(self.tmax * 1000)  # Estimate
                    node_stats['planning_times'].append(planning_time)
                    node_stats['unsolved_times'].append(planning_time)
                    iter_times.append(planning_time)
                    print(f"⏳ Task {i+1} timed out after {planning_time:.2f}s")

            # Update α (conditionally)
            if numSolved < self.NumTasksPerIterThresh:
                self.alpha = max(self.alpha - self.delta, 0.5)
                print(f"Reduced α to {self.alpha:.3f} (solved {numSolved} tasks)")

            # Train models
            print("Training models...")
            self.train_ffnn()
            _ = self.train_wunn()

            # Strict β decay
            self.beta *= self.gamma
            print(f"Decayed β to {self.beta:.6f} (γ={self.gamma:.6f})")

            # Log metrics
            avg_subopt, opt_rate, avg_time = self.compute_metrics()
            
            # Calculate statistics
            avg_nodes = np.mean(iter_nodes) if iter_nodes else 0
            solved_avg = np.mean(node_stats['nodes_per_solved']) if node_stats['nodes_per_solved'] else 0
            unsolved_avg = np.mean(node_stats['nodes_per_unsolved']) if node_stats['nodes_per_unsolved'] else 0
            avg_time_iter = np.mean(iter_times) if iter_times else 0
            
            print(f"Iteration {n} results:")
            print(f"  Solved: {numSolved}/{self.NumTasksPerIter}")
            print(f"  Suboptimality: {avg_subopt:.3f}")
            print(f"  Optimality Rate: {opt_rate:.1f}%")
            print(f"  Avg Planning Time: {avg_time_iter:.2f}s")
            print(f"  Node Generation Stats:")
            print(f"    Avg nodes this iter: {avg_nodes:.1f}")
            print(f"    Avg nodes (solved): {solved_avg:.1f}")
            print(f"    Avg nodes (unsolved): {unsolved_avg:.1f}")
            print(f"    Total nodes so far: {sum(node_stats['total_nodes'])}")

        # Final statistics report
        self.report_final_node_stats(node_stats)    

    def report_final_node_stats(self, stats):
        """Print comprehensive statistics including timing"""
        print("\n=== Final Node Generation Statistics ===")
        print(f"Total tasks attempted: {len(stats['total_nodes'])}")
        print(f"Tasks solved: {stats['tasks_solved']} ({stats['tasks_solved']/len(stats['total_nodes'])*100:.1f}%)")
        
        if self.planning_times:
            print("\nPlanning Time Statistics:")
            print(f"  Average planning time: {np.mean(self.planning_times):.2f}s")
            print(f"  Median planning time: {np.median(self.planning_times):.2f}s")
            print(f"  Min planning time: {np.min(self.planning_times):.2f}s")
            print(f"  Max planning time: {np.max(self.planning_times):.2f}s")
            print(f"  Total planning time: {sum(self.planning_times):.2f}s")
        
        if stats['nodes_per_solved']:
            print("\nSolved Tasks:")
            print(f"  Average nodes: {np.mean(stats['nodes_per_solved']):.1f}")
            print(f"  Median nodes: {np.median(stats['nodes_per_solved']):.1f}")
            print(f"  Min nodes: {np.min(stats['nodes_per_solved'])}")
            print(f"  Max nodes: {np.max(stats['nodes_per_solved'])}")
        
        if stats['nodes_per_unsolved']:
            print("\nUnsolved/Timeout Tasks:")
            print(f"  Average nodes: {np.mean(stats['nodes_per_unsolved']):.1f}")
            print(f"  Median nodes: {np.median(stats['nodes_per_unsolved']):.1f}")
        
        print(f"\nOverall Average Nodes per Task: {np.mean(stats['total_nodes']):.1f}")
        print(f"Total Nodes Generated: {sum(stats['total_nodes'])}")


# --- Example Usage ---
if __name__ == "__main__":
    goal_state = list(range(16))
    input_dim = len(encode_15puzzle_state(goal_state))

    # Algorithm parameters
    params = {
        "hidden_dim": 20,
        "dropout_rate": 0.025,
        "alpha0": 0.99,
        "beta0": 0.05,  # Will decay to 0.00001 in NumIter steps
        "epsilon": 1.0,
        "delta": 0.05,
        "kappa": 0.64,
        # gamma is now computed automatically in __init__
        "q": 0.95,
        "K": 100,
        "MaxSteps": 1000,
        "mu0": 0.0,
        "sigma2_0": 10.0,
        "NumIter": 50,  # Will decay β from 0.05 to 0.00001 in 50 steps
        "NumTasksPerIter": 10,
        "NumTasksPerIterThresh": 6,
        "TrainIter": 1000,
        "MaxTrainIter": 5000,
        "MiniBatchSize": 100,
        "tmax": 60,
        "MemoryBufferMaxRecords": 25000,
    }

    learner = LearnHeuristicPrac(input_dim, goal_state, params)
    learner.run()

    # Final metrics
    avg_subopt, opt_rate, avg_time = learner.compute_metrics()
    print(f"\n=== Final Metrics ===")
    print(f"Average Suboptimality: {avg_subopt:.3f}")
    print(f"Optimality Rate: {opt_rate:.1f}%")
    print(f"Average Planning Time: {avg_time:.2f}s")
