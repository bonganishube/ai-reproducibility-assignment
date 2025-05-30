import numpy as np
import time
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from suboptimality_v2 import LearnHeuristicPrac, get_valid_moves, encode_15puzzle_state, manhattan_distance


def run_lengthinc_experiment(length_incs=[2]):
    goal_state = list(range(16))
    input_dim = len(encode_15puzzle_state(goal_state))
    
    params = {
        "hidden_dim": 20,
        "dropout_rate": 0.025,
        "beta0": 0.05,
        "epsilon": 1.0,
        "kappa": 0.64,
        "K": 10,
        "MaxSteps": 1000,
        "mu0": 0.0,
        "sigma2_0": 10.0,
        "NumIter": 2,
        "NumTasksPerIter": 5,
        "TrainIter": 100,
        "MiniBatchSize": 32,
        "tmax": 1,
        "MemoryBufferMaxRecords": 5000,
    }
    
    results = {li: {'train': [], 'test': [], 'iter_metrics': []} for li in length_incs}
    
    for length_inc in length_incs:
        print(f"\n=== Running experiment with LengthInc={length_inc} ===")
        
        class FixedStepLearner(LearnHeuristicPrac):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.current_iter = 0
                self.iter_stats = []
            
            def generate_task(self):
                steps = length_inc * (self.current_iter + 1)
                s = generate_task_fixed_steps(self.goal_state, steps)
                return {"s": s, "sg": self.goal_state, "sigma2_e": 0}
            
            def train_ffnn(self):
                if len(self.memoryBuffer) < self.MiniBatchSize:
                    return
                
                optimizer = optim.Adam(self.nnFFNN.parameters())
                criterion = nn.MSELoss()
                
                # Convert memory buffer to tensors with proper shapes
                x_data = torch.stack(
                    [torch.tensor(x, dtype=torch.float32) for x, _ in self.memoryBuffer]
                )
                y_data = torch.tensor(
                    [y for _, y in self.memoryBuffer], dtype=torch.float32
                ).unsqueeze(1)  # Add dimension to match network output
                
                self.nnFFNN.train()
                for _ in range(self.TrainIter):
                    permutation = torch.randperm(len(x_data))
                    for i in range(0, len(x_data), self.MiniBatchSize):
                        indices = permutation[i:i + self.MiniBatchSize]
                        x_batch, y_batch = x_data[indices], y_data[indices]
                        
                        optimizer.zero_grad()
                        output = self.nnFFNN(x_batch)
                        # Ensure output and target have same shape
                        output = output.unsqueeze(1) if output.dim() == 1 else output
                        loss = criterion(output, y_batch)
                        loss.backward()
                        optimizer.step()
            
            def run_iteration(self, n):
                self.current_iter = n
                iter_solved = 0
                iter_stats = {
                    'iteration': n,
                    'tasks_generated': params["NumTasksPerIter"],
                    'tasks_solved': 0,
                    'avg_planning_time': 0,
                    'avg_nodes_expanded': 0
                }
                planning_times = []
                nodes_expanded = []
                
                for _ in range(params["NumTasksPerIter"]):
                    T = self.generate_task()
                    try:
                        start_time = time.time()
                        plan, nodes, planning_time = self.ida_star(
                            T["s"], T["sg"], self.heuristic, params["tmax"])
                        planning_times.append(planning_time)
                        nodes_expanded.append(nodes)
                        
                        if plan:
                            iter_solved += 1
                            # Store training examples
                            for state in reversed(plan[:-1]):
                                x = encode_15puzzle_state(state)
                                y = manhattan_distance(state, T["sg"])
                                self.memoryBuffer.appendleft((x, y))
                    except Exception as e:
                        print(f"Error during IDA*: {str(e)}")
                        continue
                
                iter_stats['tasks_solved'] = iter_solved
                if planning_times:
                    iter_stats['avg_planning_time'] = np.mean(planning_times)
                    iter_stats['avg_nodes_expanded'] = np.mean(nodes_expanded)
                
                self.iter_stats.append(iter_stats)
                return iter_solved
        
        learner = FixedStepLearner(input_dim, goal_state, params)
        total_train_tasks = params["NumIter"] * params["NumTasksPerIter"]
        solved_train = 0
        
        print("\nIter\tTasks\tSolved\tSolve%\tTime\tNodes")
        print("----\t-----\t------\t------\t----\t-----")
        
        for n in range(1, params["NumIter"] + 1):
            iter_solved = learner.run_iteration(n)
            solved_train += iter_solved
            
            if learner.memoryBuffer:
                learner.train_ffnn()
                if hasattr(learner, 'train_wunn'): 
                    learner.train_wunn()
            learner.beta *= learner.gamma
            
            stats = learner.iter_stats[-1]
            solve_pct = (stats['tasks_solved']/stats['tasks_generated'])*100
            print(f"{n}\t{stats['tasks_generated']}\t{stats['tasks_solved']}\t"
                  f"{solve_pct:.1f}%\t{stats['avg_planning_time']:.2f}s\t"
                  f"{stats['avg_nodes_expanded']:.0f}")
        
        train_solved_pct = (solved_train / total_train_tasks) * 100
        results[length_inc]['train'].append(train_solved_pct)
        
        # Test performance
        test_solved = 0
        test_stats = []
        for k in range(1, 101):
            test_state = generate_task_fixed_steps(goal_state, k)
            try:
                plan, nodes, planning_time = learner.ida_star(
                    test_state, goal_state, learner.heuristic, 60)
                if plan:
                    test_solved += 1
                test_stats.append({
                    'steps': k,
                    'solved': 1 if plan else 0,
                    'time': planning_time,
                    'nodes': nodes
                })
            except Exception as e:
                print(f"Error during test IDA* (k={k}): {str(e)}")
                test_stats.append({
                    'steps': k,
                    'solved': 0,
                    'time': 60,
                    'nodes': 0
                })
        
        test_solved_pct = test_solved
        results[length_inc]['test'].append(test_solved_pct)
        results[length_inc]['iter_metrics'] = learner.iter_stats
        
        print(f"\nFinal Training: {train_solved_pct:.1f}% solved")
        print(f"Final Testing: {test_solved_pct}% solved")
    
    print("\n=== FINAL COMPARISON ===")
    print("LengthInc\tTrain%\tTest%")
    print("---------\t------\t-----")
    for li in length_incs:
        print(f"LengthInc {li} data:", results[li])
        train = np.mean(results[li]['train']) if results[li]['train'] else 0
        test = np.mean(results[li]['test']) if results[li]['test'] else 0
        print(f"{li}\t\t{train:.1f}\t{test:.1f}")
    
    return results

def generate_task_fixed_steps(goal_state, steps_back):
    current_state = goal_state[:]
    for _ in range(steps_back):
        valid_moves = get_valid_moves(current_state)
        if not valid_moves:
            break
        current_state = random.choice(valid_moves)
    return current_state

# Run the experiment
if __name__ == "__main__":
    try:
        results = run_lengthinc_experiment(length_incs=[1, 2])
    except KeyboardInterrupt:
        print("\nUser stopped the experiment. Partial results:")
        print("LengthInc\tTrain%\tTest%")
        for li in results:
            train = np.mean(results[li]['train']) if results[li]['train'] else 0
            test = np.mean(results[li]['test']) if results[li]['test'] else 0
            print(f"{li}\t\t{train:.1f}\t{test:.1f}")