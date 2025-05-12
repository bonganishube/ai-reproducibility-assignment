import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import time


# 1. FFNN Model with 20 hidden neurons
class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=20, dropout_rate=0.025):
        super(FFNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Apply dropout with 2.5% rate
            nn.Linear(hidden_dim, 2)  # Outputs: ŷ and log(σ²ₐ)
        )
        self.apply(self.he_init)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def he_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            out = self.forward(torch.tensor(x, dtype=torch.float32))
        y_hat, log_sigma2_a = out[0].item(), out[1].item()
        return y_hat, np.exp(log_sigma2_a)

    def train_model(self, memory_buffer, epochs=1000):
        self.train()
        loss_fn = nn.MSELoss()
        for epoch in range(epochs):
            for x, y in memory_buffer:
                x_tensor = torch.tensor(x, dtype=torch.float32)
                y_tensor = torch.tensor([y], dtype=torch.float32)
                output = self.forward(x_tensor)
                loss = loss_fn(output[0], y_tensor)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Progress tracking: Print every 100 epochs
            if epoch % 100 == 0:
                print(f"FFNN - Epoch [{epoch}/{epochs}], Loss: {loss.item()}")


# 2. WUNN Model with 20 hidden neurons
class WUNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=20, S=5, C=1):
        super(WUNN, self).__init__()
        self.S = S
        self.C = C
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.apply(self.he_init)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def he_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward_single(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def predict_sigma_e(self, x, K):
        self.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32)
        outputs = [self.forward_single(x_tensor).item() for _ in range(K)]
        return np.var(outputs)

    def elbo_loss(self, x, y):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor([y], dtype=torch.float32)
        preds = torch.stack([self.forward_single(x_tensor) for _ in range(self.S)])
        log_likelihood = -F.mse_loss(preds.mean(), y_tensor)
        kl_div = torch.tensor(0.0)  # You can include KL term here if modeling full Bayesian NN
        beta = 0.05
        return beta * kl_div - log_likelihood

    def sample_weighted_batch(self, memory_buffer, batch_size, kappa_epsilon, K):
        weights = []
        for x, _ in memory_buffer:
            sigma2 = self.predict_sigma_e(x, K)
            if sigma2 >= kappa_epsilon:
                weights.append(np.exp(np.sqrt(sigma2)))
            else:
                weights.append(np.exp(-self.C))
        weights = np.array(weights)
        weights /= weights.sum()
        indices = np.random.choice(len(memory_buffer), size=min(batch_size, len(memory_buffer)),
                                   replace=False, p=weights)
        return [memory_buffer[i] for i in indices]

    def train_model(self, memory_buffer, max_iter, batch_size, kappa_epsilon, K):
        self.train()
        for iteration in range(max_iter):
            batch = self.sample_weighted_batch(memory_buffer, batch_size, kappa_epsilon, K)
            for x, y in batch:
                loss = self.elbo_loss(x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Progress tracking: Print every 100 iterations
            if iteration % 100 == 0:
                print(f"WUNN - Iteration [{iteration}/{max_iter}], Loss: {loss.item()}")
            
            # If convergence criteria are met, stop training
            if all(self.predict_sigma_e(x, K=1) < kappa_epsilon for x, _ in memory_buffer):
                print("Convergence reached. Stopping training.")
                break


# 3. IDA* Search for 15-Puzzle
class IDAStar:
    def __init__(self, start_state, goal_state, max_depth):
        self.start_state = start_state
        self.goal_state = goal_state
        self.max_depth = max_depth
        self.path = []
        self.cost = 0

    def _h(self, state):
        # Simple heuristic: number of misplaced tiles
        return sum([1 for i in range(len(state)) if state[i] != self.goal_state[i] and state[i] != 0])

    def _search(self, state, g, depth, path):
        f = g + self._h(state)
        if f > depth:
            return None, f  # Return None if the current path cost exceeds depth
        if state == self.goal_state:
            return path, g  # Return the path if we reached the goal
        
        min_f = float('inf')
        best_path = None

        # Explore the neighbors (valid moves)
        for move in get_valid_moves(state):
            new_path = path + [move]
            solution, f_val = self._search(move, g + 1, depth, new_path)
            if solution:
                return solution, f_val
            min_f = min(min_f, f_val)

        return None, min_f  # No solution found at this depth

    def solve(self):
        depth = self._h(self.start_state)
        while depth <= self.max_depth:
            path, _ = self._search(self.start_state, 0, depth, [self.start_state])
            if path:
                self.path = path
                return path  # Found a solution
            depth += 1
        return None  # No solution found within the depth limit


# 4. Helper Functions: Task-solving logic
def attempt_solve_task(T, nnFFNN, alpha, yq, epsilon, tmax):
    start_state = T["start"]
    goal_state = T["goal"]

    ida_star_solver = IDAStar(start_state, goal_state, max_depth=tmax)
    solution_path = ida_star_solver.solve()

    if solution_path:
        return True, solution_path  # If a solution is found, return the path
    else:
        return False, []  # If no solution found, return False and empty path


def compute_cost_to_goal(state):
    goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
    return sum([1 for i in range(len(state)) if state[i] != goal_state[i] and state[i] != 0])


# 5. GenerateTaskPrac (simulated)
def GenerateTaskPrac(nnWUNN, epsilon, MaxSteps, K, Erev, F, sg):
    s_prime = sg
    numSteps = 0
    s_prev = None

    while numSteps < MaxSteps:
        numSteps += 1
        states = {}
        for s in Erev(s_prime):
            if s_prev is not None and s == s_prev:
                continue
            x = F(s)
            sigma2_e = nnWUNN.predict_sigma_e(x, K)
            states[s] = sigma2_e

        # Sample from softmax over epistemic uncertainties
        keys = list(states.keys())
        values = np.array([states[k] for k in keys])
        probs = np.exp(values) / np.sum(np.exp(values))
        s = np.random.choice(keys, p=probs)

        if states[s] >= epsilon:
            return {"start": s, "goal": sg}  # You can expand to full T = ⟨S,O,E,C,s,sg⟩ if needed

        s_prev = s_prime
        s_prime = s

    return None

# 6. LearnHeuristicPrac (Main Training Loop)
def LearnHeuristicPrac(parameters):
    nnWUNN = WUNN(parameters["input_dim"])
    nnFFNN = FFNN(parameters["input_dim"])
    memoryBuffer = deque(maxlen=parameters["MemoryBufferMaxRecords"])
    alpha = parameters["alpha0"]
    beta = parameters["beta0"]
    updateBeta = True
    yq = float('-inf')

    for n in range(parameters["NumIter"]):
        numSolved = 0
        print(f"Iteration {n + 1}/{parameters['NumIter']} started.")
        for _ in range(parameters["NumTasksPerIter"]):
            T = GenerateTaskPrac(nnWUNN, parameters["epsilon"], parameters["MaxSteps"],
                                 parameters["K"], parameters["Erev"], parameters["F"], parameters["sg"])

            if T is None:
                continue

            plan_found, plan = attempt_solve_task(T, nnFFNN, alpha, yq, parameters["epsilon"], parameters["tmax"])
            if plan_found:
                numSolved += 1
                for s in plan:
                    if s != T["goal"]:
                        y = compute_cost_to_goal(s)
                        x = parameters["F"](s)
                        memoryBuffer.append((x, y))

        # Print progress at the end of each iteration
        print(f"Iteration {n + 1}/{parameters['NumIter']} completed. Tasks solved: {numSolved}/{parameters['NumTasksPerIter']}")

        if numSolved < parameters["NumTasksPerIterThresh"]:
            alpha = max(alpha - parameters["delta"], 0.5)
            updateBeta = False
        else:
            updateBeta = True

        nnFFNN.train_model(memoryBuffer, parameters["TrainIter"])
        nnWUNN.train_model(memoryBuffer, parameters["MaxTrainIter"],
                           parameters["MiniBatchSize"], parameters["kappa"] * parameters["epsilon"], parameters["K"])

        if updateBeta:
            beta *= parameters["gamma"]

        costs = [y for _, y in memoryBuffer]
        yq = np.quantile(costs, parameters["q"])

        # Print end of iteration summary
        print(f"End of Iteration {n + 1}/{parameters['NumIter']}. Alpha: {alpha}, Beta: {beta}")



# 7. Setup Parameters for 15-puzzle problem
parameters = {
    "NumIter": 50,
    "NumTasksPerIter": 10,
    "NumTasksPerIterThresh": 6,
    "alpha0": 0.99,
    "delta": 0.05,
    "beta0": 0.05,
    "gamma": 0.00001,
    "kappa": 0.64,
    "MemoryBufferMaxRecords": 25000,
    "TrainIter": 1000,
    "MaxTrainIter": 5000,
    "MiniBatchSize": 100,
    "epsilon": 1,
    "MaxSteps": 1000,
    "q": 0.95,
    "K": 100,
    "input_dim": 16 * 2 * 4,  # 16 tiles, each with a 2x4 grid (horizontal + vertical positions)
    "F": lambda s: np.array(encode_15puzzle_state(s)),  # 15-puzzle state encoding
    "Erev": lambda s: get_valid_moves(s),  # Function to get valid moves (this should be defined)
    "sg": 0,  # Example goal state (replace with actual)
    "tmax": 60,
}

# --- Helper Functions for 15-Puzzle Encoding and Moves ---
def encode_15puzzle_state(state):
    """Encodes the state of the 15-puzzle into a 1D array of 2x4 one-hot vectors per tile."""
    encoded = np.zeros(16 * 2 * 4)
    for i, tile in enumerate(state):
        if tile == 0:
            continue  # Skip empty tile
        # One-hot encode horizontal and vertical positions
        row, col = divmod(tile - 1, 4)
        encoded[i * 8 + row * 4 + col] = 1
    return encoded

def get_valid_moves(state):
    """Returns a list of valid moves (states) for the given 15-puzzle state."""
    zero_index = state.index(0)
    row, col = divmod(zero_index, 4)
    valid_moves = []
    # Define possible moves (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < 4 and 0 <= new_col < 4:
            new_zero_index = new_row * 4 + new_col
            new_state = state[:]
            # Swap the empty space (0) with the adjacent tile
            new_state[zero_index], new_state[new_zero_index] = new_state[new_zero_index], new_state[zero_index]
            valid_moves.append(new_state)
    return valid_moves
