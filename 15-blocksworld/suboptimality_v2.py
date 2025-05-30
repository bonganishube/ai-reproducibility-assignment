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

# Define the PDB patterns for 15-blocksworld (12 4-block PDBs)
PDB_PATTERNS = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [12, 13, 14, 15],
    [3, 4, 5, 6],
    [7, 8, 9, 10],
    [11, 12, 13, 14],
    [1, 2, 14, 15],
    [2, 3, 4, 5],
    [6, 7, 8, 9],
    [10, 11, 12, 13],
    [1, 13, 14, 15]
]

def count_out_of_place_blocks(state, goal_state):
    return sum(1 for block in state if state[block] != goal_state[block])

def count_stacks(state):
    # A stack is defined by a block with nothing on top of it
    stacked_blocks = set()
    for block, on_top in state.items():
        if on_top is not None:
            stacked_blocks.add(on_top)
    return len(state) - len(stacked_blocks)

def compute_pdb_heuristic(state, goal_state, pattern):
    # Simplified PDB heuristic for blocksworld
    distance = 0
    for block in pattern:
        # Count blocks above current position
        current_pos = state[block]
        above = current_pos
        while above is not None:
            if above in pattern:
                distance += 1
            above = state.get(above, None)
        
        # Count blocks above goal position
        goal_pos = goal_state[block]
        above = goal_pos
        while above is not None:
            if above in pattern:
                distance += 1
            above = goal_state.get(above, None)
    return distance

def encode_blocksworld_state(state, goal_state):
    features = np.zeros(14)
    
    # Compute PDB features (f1-f12)
    for i, pattern in enumerate(PDB_PATTERNS):
        features[i] = compute_pdb_heuristic(state, goal_state, pattern)
    
    features[12] = count_out_of_place_blocks(state, goal_state)  # f13
    features[13] = count_stacks(state)  # f14
    
    return features

def get_valid_moves_blocksworld(state):
    valid_moves = []
    # Find all blocks that can be moved (those with nothing on top)
    movable_blocks = [block for block in state if all(top != block for top in state.values())]
    
    for block in movable_blocks:
        # Can move to table (if not already there)
        if state[block] is not None:
            new_state = state.copy()
            new_state[block] = None
            valid_moves.append(new_state)
        
        # Can move onto other blocks (that aren't this block)
        for other_block in state:
            if other_block != block and state[other_block] != block:
                new_state = state.copy()
                new_state[block] = other_block
                valid_moves.append(new_state)
    
    return valid_moves

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_mu=0.0, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_logvar = nn.Parameter(torch.empty(out_features))
        nn.init.normal_(self.weight_mu, mean=prior_mu, std=prior_sigma / 10)
        nn.init.constant_(self.weight_logvar, math.log(prior_sigma**2))
        nn.init.normal_(self.bias_mu, mean=prior_mu, std=prior_sigma / 10)
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
            1 + self.weight_logvar - math.log(self.prior_sigma**2)
        )
        kl_bias = 0.5 * (
            (self.bias_mu - self.prior_mu).pow(2) + torch.exp(self.bias_logvar)
        ) / (self.prior_sigma**2) - 0.5 * (
            1 + self.bias_logvar - math.log(self.prior_sigma**2)
        )
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
    def __init__(self, input_dim, hidden_dim=8, dropout_rate=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze()

class LearnHeuristicPrac:
    def __init__(self, input_dim, goal_state, params):
        self.nnWUNN = WUNN(
            input_dim,
            params["hidden_dim"],
            prior_mu=params["mu0"],
            prior_sigma=math.sqrt(params["sigma2_0"]),
        )
        self.nnFFNN = FFNN(input_dim, params["hidden_dim"], params["dropout_rate"])
        self.beta = params["beta0"]
        self.epsilon = params["epsilon"]
        self.kappa = params["kappa"]
        self.gamma = (0.00001 / params["beta0"]) ** (1 / params["NumIter"])
        self.K = params["K"]
        self.max_steps = params["MaxSteps"]
        self.NumIter = params["NumIter"]
        self.NumTasksPerIter = params["NumTasksPerIter"]
        self.TrainIter = params["TrainIter"]
        self.MiniBatchSize = params["MiniBatchSize"]
        self.tmax = params["tmax"]
        self.memoryBuffer = deque(maxlen=params["MemoryBufferMaxRecords"])
        self.planner_costs = []
        self.optimal_costs = []
        self.planning_times = []
        self.optimal_solutions_count = 0
        self.total_solved_tasks = 0
        self.goal_state = goal_state

    def max_admissible_heuristic(self, state):
        x = encode_blocksworld_state(state, self.goal_state)
        admissible_values = x[:12]  # f1-f12 are admissible
        return np.max(admissible_values)

    def generate_task(self):
        # Generate a random blocksworld task
        blocks = list(self.goal_state.keys())
        
        # Start from goal state and perform random moves
        state = self.goal_state.copy()
        moves = 0
        max_moves = 30  # Maximum number of moves to scramble
        
        while moves < max_moves:
            valid_moves = get_valid_moves_blocksworld(state)
            if not valid_moves:
                break  # No valid moves available
            
            state = random.choice(valid_moves)
            moves += 1
        
        return {
            "s": state,
            "sg": self.goal_state,
            "sigma2_e": 1.0  # Initial uncertainty
        }

    def ida_star(self, start, goal, heuristic, tmax, start_time):
        threshold = heuristic(start)
        path = [start]
        total_nodes = 0
        
        def search(g, bound):
            nonlocal total_nodes
            if time.time() - start_time > tmax:
                raise TimeoutError()
            
            node = path[-1]
            f = g + heuristic(node)
            if f > bound:
                return f
            if node == goal:
                return True
            
            min_t = float('inf')
            neighbors = get_valid_moves_blocksworld(node)
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

        planning_start = time.time()
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

    def heuristic(self, state):
        x = encode_blocksworld_state(state, self.goal_state)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        self.nnFFNN.eval()
        with torch.no_grad():
            y_hat = self.nnFFNN(x_tensor).item()
        return max(y_hat, self.max_admissible_heuristic(state))

    def train_ffnn(self):
        if len(self.memoryBuffer) < self.MiniBatchSize:
            return

        optimizer = optim.Adam(self.nnFFNN.parameters())
        criterion = nn.MSELoss()
        x_data = torch.stack(
            [torch.tensor(x, dtype=torch.float32) for x, _ in self.memoryBuffer]
        )
        y_data = torch.tensor(
            [y for _, y in self.memoryBuffer], dtype=torch.float32
        )

        self.nnFFNN.train()
        for _ in range(self.TrainIter):
            permutation = torch.randperm(len(x_data))
            for i in range(0, len(x_data), self.MiniBatchSize):
                indices = permutation[i : i + self.MiniBatchSize]
                x_batch, y_batch = x_data[indices], y_data[indices]
                optimizer.zero_grad()
                output = self.nnFFNN(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

    def train_wunn(self):
        if len(self.memoryBuffer) < self.MiniBatchSize:
            return False

        self.nnWUNN.train()
        optimizer = optim.Adam(self.nnWUNN.parameters(), lr=0.01)
        early_stop = False
        uncertainties = []
        
        for x, _ in self.memoryBuffer:
            sigma2_e = max(self.nnWUNN.predict_sigma_e(x, K=10), 1e-8)
            uncertainties.append(sigma2_e)
        
        max_uncertainty = max(uncertainties) if uncertainties else 1.0
        if max_uncertainty > 1e8:
            scale_factor = max_uncertainty / 1e8
            uncertainties = [u/scale_factor for u in uncertainties]
        
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
        
        max_weight = max(weights) if weights else 1.0
        if max_weight > 1e6:
            weights = [w/max_weight*1e6 for w in weights]
        
        for iter in range(100):
            batch_indices = random.choices(
                range(len(self.memoryBuffer)),
                weights=weights,
                k=min(self.MiniBatchSize, len(self.memoryBuffer)))
            batch = [self.memoryBuffer[i] for i in batch_indices]

            total_loss = 0
            for x, y in batch:
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                y_tensor = torch.tensor([y], dtype=torch.float32).unsqueeze(1)
                
                preds = torch.stack(
                    [self.nnWUNN.forward_single(x_tensor) for _ in range(self.nnWUNN.S)]
                )
                pred_mean = preds.mean(dim=0)
                
                log_likelihood = -F.mse_loss(pred_mean, y_tensor)
                kl_div = self.nnWUNN.fc1.kl_divergence() + self.nnWUNN.fc2.kl_divergence()
                loss = self.beta * kl_div - log_likelihood
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if iter % 10 == 0:
                if all(s < self.kappa * self.epsilon for s in uncertainties[:100]):
                    early_stop = True
                    break

        return early_stop

    def run(self):
        print("Iter\tTime\tSubopt%\tOpt%\tGenerated")

        for n in range(1, self.NumIter + 1):
            iter_planner_costs = []
            iter_optimal_costs = []
            iter_solved_count = 0
            iter_times = []
            iter_nodes = []
            
            for _ in range(self.NumTasksPerIter):
                try:
                    T = self.generate_task()
                    if not T:
                        continue

                    start_time = time.time()
                    plan, nodes_generated, planning_time = self.ida_star(
                        T["s"], T["sg"], 
                        self.heuristic,
                        self.tmax, start_time
                    )
                    
                    iter_times.append(planning_time)
                    iter_nodes.append(nodes_generated)
                    
                    if plan:
                        plan_cost = len(plan) - 1
                        optimal_cost = count_out_of_place_blocks(T["s"], T["sg"])
                        iter_planner_costs.append(plan_cost)
                        iter_optimal_costs.append(optimal_cost)
                        iter_solved_count += 1
                        self.planner_costs.append(plan_cost)
                        self.optimal_costs.append(optimal_cost)
                        self.total_solved_tasks += 1
                        if plan_cost == optimal_cost:
                            self.optimal_solutions_count += 1
                        
                        for state in reversed(plan[:-1]):
                            x = encode_blocksworld_state(state, T["sg"])
                            y = count_out_of_place_blocks(state, T["sg"])
                            self.memoryBuffer.appendleft((x, y))
                            
                except Exception as e:
                    print(f"Error in task: {str(e)}")
                    continue

            self.train_ffnn()
            self.train_wunn()
            self.beta *= self.gamma

            if iter_solved_count > 0:
                subopt_pct = [((y/y_star)-1)*100 for y, y_star in zip(iter_planner_costs, iter_optimal_costs)]
                avg_subopt_pct = sum(subopt_pct)/len(subopt_pct) if subopt_pct else 0
                opt_count = sum(1 for y, y_star in zip(iter_planner_costs, iter_optimal_costs) if y == y_star)
                opt_rate = (opt_count/iter_solved_count)*100 if iter_solved_count > 0 else 0
            else:
                avg_subopt_pct = 0
                opt_rate = 0
                
            avg_time_iter = np.mean(iter_times) if iter_times else 0
            avg_nodes_iter = np.mean(iter_nodes) if iter_nodes else 0
            
            print(f"{n}\t{avg_time_iter:.2f}\t{avg_subopt_pct:.1f}%\t{opt_rate:.1f}%\t{avg_nodes_iter:.0f}")

        if self.total_solved_tasks > 0:
            global_subopt = sum((y/y_star-1)*100 for y, y_star in zip(self.planner_costs, self.optimal_costs)) / self.total_solved_tasks
            global_opt_rate = self.optimal_solutions_count / self.total_solved_tasks * 100
        else:
            global_subopt = 0
            global_opt_rate = 0

        print("\n=== Final Results ===")
        print(f"Solved: {self.total_solved_tasks}/{self.NumIter*self.NumTasksPerIter}")
        print(f"Suboptimality: {global_subopt:.1f}%")
        print(f"Optimal solutions: {global_opt_rate:.1f}%")

if __name__ == "__main__":
    # Define goal state for 15-blocksworld: all blocks in one stack
    goal_state = {
        1: None,   # Block 1 is on table
        2: 1,       # Block 2 is on block 1
        3: 2,       # Block 3 is on block 2
        4: 3,       # ...
        5: 4,
        6: 5,
        7: 6,
        8: 7,
        9: 8,
        10: 9,
        11: 10,
        12: 11,
        13: 12,
        14: 13,
        15: 14      # Block 15 is on block 14
    }
    
    input_dim = 14  # Number of features in our encoding
    
    params = {
        "hidden_dim": 8,
        "dropout_rate": 0.0,
        "beta0": 0.05,
        "epsilon": 1.0,
        "kappa": 0.64,
        "K": 10,
        "MaxSteps": 1000,
        "mu0": 0.0,
        "sigma2_0": 10.0,
        "NumIter": 75,
        "NumTasksPerIter": 5,
        "TrainIter": 100,
        "MiniBatchSize": 32,
        "tmax": 10,  # 10 seconds per task
        "MemoryBufferMaxRecords": 5000,
    }

    learner = LearnHeuristicPrac(input_dim, goal_state, params)
    learner.run()