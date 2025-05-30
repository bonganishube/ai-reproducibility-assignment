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

def encode_15puzzle_state(state):
    encoded = np.zeros(16 * 2 * 4)
    for tile in range(16):
        idx = state.index(tile)
        row, col = divmod(idx, 4)
        encoded[tile * 8 + row] = 1
        encoded[tile * 8 + 4 + col] = 1
    return encoded

def get_valid_moves(state):
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
    total = 0
    for i in range(1, 16):
        curr_idx = state.index(i)
        goal_idx = goal_state.index(i)
        curr_row, curr_col = divmod(curr_idx, 4)
        goal_row, goal_col = divmod(goal_idx, 4)
        total += abs(curr_row - goal_row) + abs(curr_col - goal_col)
    return total

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
    def __init__(self, input_dim, hidden_dim=20, S=5, prior_mu=0.0, prior_sigma=1.0):
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
    def __init__(self, input_dim, hidden_dim=20, dropout_rate=0.025):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)
        mean = output[:, 0]
        var = F.softplus(output[:, 1])
        return mean, var

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            mean, var = self.forward(x)
            return mean.item(), var.item()

class LearnHeuristicPrac:
    def __init__(self, input_dim, goal_state, params):
        self.nnWUNN = WUNN(
            input_dim,
            params["hidden_dim"],
            prior_mu=params["mu0"],
            prior_sigma=math.sqrt(params["sigma2_0"]),
        )
        self.nnFFNN = FFNN(input_dim, params["hidden_dim"], params["dropout_rate"])        
        self.alpha = params["alpha0"]
        self.beta = params["beta0"]
        self.epsilon = params["epsilon"]
        self.delta = params["delta"]
        self.kappa = params["kappa"]
        self.gamma = (0.00001 / params["beta0"]) ** (1 / params["NumIter"])
        self.q = params["q"]
        self.K = params["K"]
        self.max_steps = params["MaxSteps"]
        self.NumIter = params["NumIter"]
        self.NumTasksPerIter = params["NumTasksPerIter"]
        self.NumTasksPerIterThresh = params["NumTasksPerIterThresh"]
        self.TrainIter = params["TrainIter"]
        self.MaxTrainIter = params["MaxTrainIter"]
        self.MiniBatchSize = params["MiniBatchSize"]
        self.tmax = params["tmax"]
        self.memoryBuffer = deque(maxlen=params["MemoryBufferMaxRecords"])
        self.planner_costs = []
        self.optimal_costs = []
        self.planning_times = []
        self.suboptimalities = []
        self.optimal_solutions_count = 0
        self.total_solved_tasks = 0
        self.optimality_counts = 0
        self.goal_state = goal_state

    def h(self, alpha, mu, sigma):
        return mu + sigma * norm.ppf(alpha)

    def generate_task(self):
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
                states[tuple(s)] = sigma2_e

            if not states:
                break

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
        threshold = heuristic(start)
        path = [start]
        total_nodes = 0
        
        def search(g, bound):
            nonlocal total_nodes
            if time.time() - planning_start > tmax:
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

    def uncertainty_aware_heuristic(self, state):
        x = encode_15puzzle_state(state)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        self.nnFFNN.eval()
        with torch.no_grad():
            mean, logvar = self.nnFFNN(x_tensor)
            y_hat = mean.item()
            sigma2_a = torch.exp(logvar).item()
        sigma2_t = sigma2_a if y_hat < self.yq else self.epsilon
        h_val = self.h(self.alpha, y_hat, math.sqrt(sigma2_t))
        return max(h_val, 0)

    def compute_metrics(self):
        if not self.planner_costs:
            return 0.0, 0.0, 0.0

        suboptimalities = [
            (y / y_star) - 1
            for y, y_star in zip(self.planner_costs, self.optimal_costs)
            if y_star > 0
        ]
        avg_suboptimality = sum(suboptimalities) / len(suboptimalities) if suboptimalities else 0.0

        optimal_percentage = (
            (self.optimal_solutions_count / self.total_solved_tasks * 100) 
            if self.total_solved_tasks > 0 
            else 0.0
        )

        avg_planning_time = (
            sum(self.planning_times) / len(self.planning_times) 
            if self.planning_times 
            else 0.0
        )

        return avg_suboptimality, optimal_percentage, avg_planning_time

    def train_ffnn(self):
        if len(self.memoryBuffer) < self.MiniBatchSize:
            return

        optimizer = optim.Adam(self.nnFFNN.parameters())
        criterion = nn.GaussianNLLLoss()
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
                mean, logvar = self.nnFFNN(x_batch)
                loss = criterion(mean, y_batch, torch.exp(logvar))
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
            sigma2_e = self.nnWUNN.predict_sigma_e(x, K=10)
            uncertainties.append(sigma2_e)

        weights = []
        for sigma2_e in uncertainties:
            if sigma2_e >= self.kappa * self.epsilon:
                weight = math.exp(math.sqrt(sigma2_e))
            else:
                weight = math.exp(-1)
            weights.append(weight)

        for iter in range(self.MaxTrainIter):
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

            batch_indices = random.choices(
                range(len(self.memoryBuffer)),
                weights=weights,
                k=min(self.MiniBatchSize, len(self.memoryBuffer))
            )
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

        return early_stop

    def run(self):
        self.yq = -np.inf
        all_alphas = []
        all_suboptimalities = []
        all_optimal_rates = []
        all_planning_times = []
        all_nodes_generated = []
        self.planner_costs = []
        self.optimal_costs = []
        self.optimal_solutions_count = 0
        self.total_solved_tasks = 0

        print("Iter\tα\tTime\tSubopt%\tOpt%\tGenerated")

        for n in range(1, self.NumIter + 1):
            iter_planner_costs = []
            iter_optimal_costs = []
            iter_solved_count = 0
            iter_times = []
            iter_nodes = []
            
            for i in range(self.NumTasksPerIter):
                T = self.generate_task()
                if not T:
                    continue

                try:
                    start_time = time.time()
                    plan, nodes_generated, planning_time = self.ida_star(
                        T["s"], T["sg"], 
                        self.uncertainty_aware_heuristic,
                        self.tmax, start_time
                    )
                    
                    iter_times.append(planning_time)
                    iter_nodes.append(nodes_generated)
                    
                    if plan:
                        plan_cost = len(plan) - 1
                        optimal_cost = manhattan_distance(T["s"], T["sg"])
                        iter_planner_costs.append(plan_cost)
                        iter_optimal_costs.append(optimal_cost)
                        iter_solved_count += 1
                        self.planner_costs.append(plan_cost)
                        self.optimal_costs.append(optimal_cost)
                        self.total_solved_tasks += 1
                        if plan_cost == optimal_cost:
                            self.optimal_solutions_count += 1
                        
                        for state in reversed(plan[:-1]):
                            x = encode_15puzzle_state(state)
                            y = manhattan_distance(state, T["sg"])
                            self.memoryBuffer.appendleft((x, y))
                            
                except TimeoutError:
                    planning_time = time.time() - start_time
                    iter_times.append(planning_time)
                    iter_nodes.append(self.tmax * 1000)

            if iter_solved_count < self.NumTasksPerIterThresh:
                self.alpha = max(self.alpha - self.delta, self.alpha)

            self.train_ffnn()
            _ = self.train_wunn()
            self.beta *= self.gamma

            if iter_solved_count > 0:
                subopt_pct = [((y/y_star)-1)*100 for y, y_star in zip(iter_planner_costs, iter_optimal_costs)]
                avg_subopt_pct = sum(subopt_pct)/len(subopt_pct)
                opt_count = sum(1 for y, y_star in zip(iter_planner_costs, iter_optimal_costs) if y == y_star)
                opt_rate = (opt_count/iter_solved_count)*100
            else:
                avg_subopt_pct = 0
                opt_rate = 0
                
            avg_time_iter = np.mean(iter_times) if iter_times else 0
            avg_nodes_iter = np.mean(iter_nodes) if iter_nodes else 0
            
            all_alphas.append(self.alpha)
            if iter_solved_count > 0:
                all_suboptimalities.append(avg_subopt_pct)
                all_optimal_rates.append(opt_rate)
            all_planning_times.append(avg_time_iter)
            all_nodes_generated.append(avg_nodes_iter)
            
            print(f"{n}\t{self.alpha:.3f}\t{avg_time_iter:.2f}\t{avg_subopt_pct:.1f}%\t{opt_rate:.1f}%\t{avg_nodes_iter:.0f}")

        final_avg_alpha = np.mean(all_alphas)
        final_avg_time = np.mean(all_planning_times)
        final_avg_nodes = np.mean(all_nodes_generated)
        valid_subopts = [s for s in all_suboptimalities if s != 0]
        final_avg_subopt = np.mean(valid_subopts) if valid_subopts else 0
        valid_opts = [o for o in all_optimal_rates if o != 0]
        final_avg_opt = np.mean(valid_opts) if valid_opts else 0

        if self.total_solved_tasks > 0:
            global_subopt_pct = [((y/y_star)-1)*100 for y, y_star in zip(self.planner_costs, self.optimal_costs)]
            global_avg_subopt = sum(global_subopt_pct)/len(global_subopt_pct)
            global_opt_rate = (self.optimal_solutions_count/self.total_solved_tasks)*100
        else:
            global_avg_subopt = 0
            global_opt_rate = 0

        print("\n=== Final Averages ===")
        print("Iter\tα\tTime\tSubopt%\tOpt%\tGenerated")
        print(f"All\t{final_avg_alpha:.3f}\t{final_avg_time:.2f}\t{final_avg_subopt:.1f}%\t{final_avg_opt:.1f}%\t{final_avg_nodes:.0f}")

if __name__ == "__main__":
    goal_state = list(range(16))
    input_dim = len(encode_15puzzle_state(goal_state))
    params = {
        "hidden_dim": 20,
        "dropout_rate": 0.025,
        "alpha0": 0.99,
        "beta0": 0.05,
        "epsilon": 1.0,
        "delta": 0.05,
        "kappa": 0.64,
        "q": 0.95,
        "K": 10,
        "MaxSteps": 1000,
        "mu0": 0.0,
        "sigma2_0": 10.0,
        "NumIter": 50,
        "NumTasksPerIter": 5,
        "NumTasksPerIterThresh": 6,
        "TrainIter": 100,
        "MaxTrainIter": 500,
        "MiniBatchSize": 32,
        "tmax": 10,
        "MemoryBufferMaxRecords": 5000,
    }

    learner = LearnHeuristicPrac(input_dim, goal_state, params)
    learner.run()