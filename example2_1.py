import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
import time
from scipy.stats import norm
from collections import deque

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
            new_state[zero_index], new_state[new_zero_index] = new_state[new_zero_index], new_state[zero_index]
            valid_moves.append(new_state)
    return valid_moves

def scramble(state, steps=15):
    """Scrambles the puzzle by making random moves"""
    s = state[:]
    for _ in range(steps):
        s = random.choice(get_valid_moves(s))
    return s

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
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize parameters properly
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_logvar = nn.Parameter(torch.empty(out_features))
        
        # Initialize with proper values
        nn.init.normal_(self.weight_mu, mean=0, std=0.1)
        nn.init.constant_(self.weight_logvar, -3)
        nn.init.normal_(self.bias_mu, mean=0, std=0.1)
        nn.init.constant_(self.bias_logvar, -3)
        
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
        kl = 0.5 * (self.weight_mu.pow(2) + torch.exp(self.weight_logvar) - self.weight_logvar - 1).sum()
        kl += 0.5 * (self.bias_mu.pow(2) + torch.exp(self.bias_logvar) - self.bias_logvar - 1).sum()
        return kl

class WUNN(nn.Module):
    """Weight Uncertainty Neural Network for epistemic uncertainty"""
    def __init__(self, input_dim, hidden_dim=20, S=5, prior_std=1.0):
        super().__init__()
        self.S = S  # Number of samples for forward pass
        self.fc1 = BayesianLinear(input_dim, hidden_dim, prior_std)
        self.fc2 = BayesianLinear(hidden_dim, 1, prior_std)
        
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
    def __init__(self, input_dim, hidden_dim=20, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, 1)
        self.fc2_var = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_normal_(self.fc2_mean.weight)
        nn.init.xavier_normal_(self.fc2_var.weight)
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
        # Initialize models
        self.nnWUNN = WUNN(input_dim, params['hidden_dim'])
        self.nnFFNN = FFNN(input_dim, params['hidden_dim'], params['dropout_rate'])
        
        # Algorithm parameters
        self.alpha = params['alpha0']
        self.beta = params['beta0']
        self.epsilon = params['epsilon']
        self.delta = params['delta']
        self.kappa = params['kappa']
        self.gamma = params['gamma']
        self.q = params['q']
        self.K = params['K']
        
        # Training parameters
        self.NumIter = params['NumIter']
        self.NumTasksPerIter = params['NumTasksPerIter']
        self.NumTasksPerIterThresh = params['NumTasksPerIterThresh']
        self.TrainIter = params['TrainIter']
        self.MaxTrainIter = params['MaxTrainIter']
        self.MiniBatchSize = params['MiniBatchSize']
        self.tmax = params['tmax']
        
        # Memory buffer (using deque for efficient trimming)
        self.memoryBuffer = deque(maxlen=params['MemoryBufferMaxRecords'])
        
        # Metrics tracking
        self.planner_costs = []
        self.optimal_costs = []
        self.suboptimalities = []
        self.optimality_counts = 0
        self.goal_state = goal_state
        
    def h(self, alpha, mu, sigma):
        """Quantile function of normal distribution"""
        return mu + sigma * norm.ppf(alpha)
    
    def generate_task(self):
        """Generates a task with high epistemic uncertainty"""
        start_state = scramble(self.goal_state)
        x = encode_15puzzle_state(start_state)
        sigma2_e = self.nnWUNN.predict_sigma_e(x, self.K)
        
        if sigma2_e >= self.epsilon:
            return {
                's': start_state,
                'sg': self.goal_state,
                'sigma2_e': sigma2_e
            }
        return None
    
    def ida_star(self, start, goal, heuristic, tmax, start_time):
        """IDA* implementation with time limit"""
        threshold = heuristic(start)
        path = [start]
        
        def search(g, bound):
            if time.time() - start_time > tmax:
                raise TimeoutError()
            
            node = path[-1]
            f = g + heuristic(node)
            if f > bound:
                return f
            if node == goal:
                return True
            
            min_t = float('inf')
            for neighbor in get_valid_moves(node):
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
        
        while True:
            t = search(0, threshold)
            if t is True:
                return path
            if t == float('inf'):
                return None
            threshold = t
    
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
        """Compute suboptimality and optimality metrics"""
        if not self.planner_costs:
            return 0.0, 0.0
        
        # Calculate suboptimality (u_i)
        suboptimalities = [
            (y / y_star) - 1 
            for y, y_star in zip(self.planner_costs, self.optimal_costs)
            if y_star > 0  # Avoid division by zero
        ]
        avg_suboptimality = sum(suboptimalities) / len(suboptimalities) if suboptimalities else 0.0

        # Calculate optimality rate (% tasks solved optimally)
        optimality_rate = (self.optimality_counts / len(self.planner_costs)) * 100 if self.planner_costs else 0.0

        return avg_suboptimality, optimality_rate
    
    def train_ffnn(self):
        """Trains FFNN on entire memory buffer"""
        if len(self.memoryBuffer) < self.MiniBatchSize:
            return
        
        optimizer = optim.Adam(self.nnFFNN.parameters())
        criterion = nn.GaussianNLLLoss()
        
        # Convert memory buffer to tensors
        x_data = torch.stack([torch.tensor(x, dtype=torch.float32) for x, _ in self.memoryBuffer])
        y_data = torch.tensor([y for _, y in self.memoryBuffer], dtype=torch.float32).unsqueeze(1)
        
        self.nnFFNN.train()
        for _ in range(self.TrainIter):
            # Shuffle and batch the data
            permutation = torch.randperm(len(x_data))
            for i in range(0, len(x_data), self.MiniBatchSize):
                indices = permutation[i:i+self.MiniBatchSize]
                x_batch, y_batch = x_data[indices], y_data[indices]
                
                optimizer.zero_grad()
                mean, logvar = self.nnFFNN(x_batch)
                loss = criterion(mean, y_batch, torch.exp(logvar))
                loss.backward()
                optimizer.step()
    
    def train_wunn(self):
        """Trains WUNN with early stopping condition"""
        if len(self.memoryBuffer) < self.MiniBatchSize:
            return False
        
        self.nnWUNN.train()
        optimizer = optim.Adam(self.nnWUNN.parameters())
        
        early_stop = False
        for iter in range(self.MaxTrainIter):
            # Mini-batch training
            batch = random.sample(self.memoryBuffer, self.MiniBatchSize)
            total_loss = 0
            
            for x, y in batch:
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                y_tensor = torch.tensor([y], dtype=torch.float32)
                
                # Forward pass with multiple samples
                preds = torch.stack([self.nnWUNN.forward_single(x_tensor) for _ in range(self.nnWUNN.S)])
                
                # Compute loss
                log_likelihood = -F.mse_loss(preds.mean(), y_tensor)
                kl_div = self.nnWUNN.fc1.kl_divergence() + self.nnWUNN.fc2.kl_divergence()
                loss = self.beta * kl_div - log_likelihood
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Check early stopping condition periodically
            if iter % 10 == 0:
                self.nnWUNN.eval()
                all_low_uncertainty = True
                
                # Check a subset of the buffer for efficiency
                check_samples = min(100, len(self.memoryBuffer))
                for x, _ in random.sample(self.memoryBuffer, check_samples):
                    sigma2_e = self.nnWUNN.predict_sigma_e(x, 10)  # Smaller K for faster checking
                    if sigma2_e >= self.kappa * self.epsilon:
                        all_low_uncertainty = False
                        break
                
                if all_low_uncertainty:
                    print(f"WUNN training stopped early (iteration {iter})")
                    early_stop = True
                    break
                
                self.nnWUNN.train()
        
        return early_stop
    
    def run(self):
        """Main learning loop"""
        # Initialize yq
        self.yq = -np.inf
        
        for n in range(1, self.NumIter + 1):
            print(f"\n=== Iteration {n}/{self.NumIter} ===")
            
            # Update yq from memory buffer
            if self.memoryBuffer:
                costs = [y for _, y in self.memoryBuffer]
                self.yq = np.quantile(costs, self.q)
            print(f"Current yq (q={self.q} quantile): {self.yq:.2f}, α: {self.alpha:.3f}, β: {self.beta:.3f}")
            
            # Generate and solve tasks
            numSolved = 0
            for i in range(self.NumTasksPerIter):
                T = self.generate_task()
                if T is None:
                    print("No task generated (low uncertainty)")
                    continue
                
                try:
                    start_time = time.time()
                    plan = self.ida_star(
                        T['s'],
                        T['sg'],
                        self.uncertainty_aware_heuristic,
                        self.tmax,
                        start_time
                    )
                    
                    if plan:
                        numSolved += 1
                        plan_cost = len(plan) - 1  # Cost = steps taken
                        optimal_cost = manhattan_distance(T['s'], T['sg'])
                        
                        # Record costs for metrics
                        self.planner_costs.append(plan_cost)
                        self.optimal_costs.append(optimal_cost)
                        
                        # Track optimal solutions
                        if plan_cost == optimal_cost:
                            self.optimality_counts += 1
                        
                        print(f"✓ Solved task {i+1} (cost: {plan_cost}, optimal: {optimal_cost})")
                        
                        # Add to memory buffer (most recent first)
                        for state in reversed(plan[:-1]):  # Exclude goal state, reversed for recent first
                            x = encode_15puzzle_state(state)
                            y = manhattan_distance(state, T['sg'])
                            self.memoryBuffer.appendleft((x, y))  # Add to front
                
                except TimeoutError:
                    print(f"⏳ Task {i+1} timed out")
                    continue
            
            # Update α and β based on solved tasks
            if numSolved < self.NumTasksPerIterThresh:
                self.alpha = max(self.alpha - self.delta, 0.5)
                self.updateBeta = False
                print(f"Reduced α to {self.alpha:.3f} (only solved {numSolved} tasks)")
            else:
                self.updateBeta = True
            
            # Train models
            print("Training models...")
            self.train_ffnn()
            early_stop = self.train_wunn()
            
            # Update β if conditions met
            if self.updateBeta and not early_stop:
                self.beta *= self.gamma
                print(f"Reduced β to {self.beta:.3f}")
            
            # Compute and log metrics
            avg_subopt, opt_rate = self.compute_metrics()
            print(f"Iteration {n} complete:")
            print(f"  Solved {numSolved}/{self.NumTasksPerIter} tasks")
            print(f"  Avg Suboptimality: {avg_subopt:.3f}")
            print(f"  Optimality Rate: {opt_rate:.1f}%")

# --- Example Usage ---
if __name__ == "__main__":
    # Define goal state and feature dimension
    goal_state = list(range(16))
    input_dim = len(encode_15puzzle_state(goal_state))
    
    # Algorithm parameters
    params = {
        'hidden_dim': 20,
        'dropout_rate': 0.1,
        'alpha0': 0.99,
        'beta0': 0.05,
        'epsilon': 1.0,
        'delta': 0.01,
        'kappa': 0.5,
        'gamma': 0.9,
        'q': 0.95,
        'K': 50,
        'NumIter': 30,
        'NumTasksPerIter': 25,
        'NumTasksPerIterThresh': 3,
        'TrainIter': 200,
        'MaxTrainIter': 1000,
        'MiniBatchSize': 32,
        'tmax': 30,  # seconds per task
        'MemoryBufferMaxRecords': 5000
    }
    
    # Initialize and run the algorithm
    learner = LearnHeuristicPrac(input_dim, goal_state, params)
    learner.run()

    # Final metrics
    avg_subopt, opt_rate = learner.compute_metrics()
    print(f"\n=== Final Metrics ===")
    print(f"Average Suboptimality: {avg_subopt:.3f}")
    print(f"Optimality Rate: {opt_rate:.1f}%")