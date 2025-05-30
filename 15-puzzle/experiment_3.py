import numpy as np
import time
import random
from collections import defaultdict
from experiment_2 import LearnHeuristicPrac, get_valid_moves, encode_15puzzle_state, manhattan_distance


def run_lengthinc_experiment():
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
        "NumIter": 20,
        "NumTasksPerIter": 5,
        "TrainIter": 100,
        "MiniBatchSize": 32,
        "tmax": 5,  # Increased from 1s to 5s for better results
        "MemoryBufferMaxRecords": 5000,
    }
    
    length_incs = [1, 2, 4, 6, 8, 10]
    results = {li: {'train': [], 'test': []} for li in length_incs}
    
    for length_inc in length_incs:
        print(f"\n=== Running experiment with LengthInc={length_inc} ===")
        
        class FixedStepLearner(LearnHeuristicPrac):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.current_iter = 0
            
            def generate_task(self):
                steps = length_inc * (self.current_iter + 1)
                s = generate_task_fixed_steps(self.goal_state, steps)
                return {"s": s, "sg": self.goal_state, "sigma2_e": 0}
        
        learner = FixedStepLearner(input_dim, goal_state, params)
        total_train_tasks = params["NumIter"] * params["NumTasksPerIter"]
        solved_train = 0
        
        for n in range(1, params["NumIter"] + 1):
            learner.current_iter = n
            iter_solved = 0
            
            for _ in range(params["NumTasksPerIter"]):
                T = learner.generate_task()
                try:
                    plan, _, _ = learner.ida_star(T["s"], T["sg"], learner.heuristic, params["tmax"])
                    if plan:
                        iter_solved += 1
                        # Store training examples
                        for state in reversed(plan[:-1]):
                            x = encode_15puzzle_state(state)
                            y = manhattan_distance(state, T["sg"])
                            learner.memoryBuffer.appendleft((x, y))
                except Exception as e:
                    continue
            
            solved_train += iter_solved
            if learner.memoryBuffer:  # Only train if we have data
                learner.train_ffnn()
                learner.train_wunn()
            learner.beta *= learner.gamma
        
        # Calculate training solve rate
        train_solved_pct = (solved_train / total_train_tasks) * 100
        results[length_inc]['train'].append(train_solved_pct)
        
        # Test performance
        test_solved = 0
        for k in range(1, 101):  # 100 test tasks
            test_state = generate_task_fixed_steps(goal_state, k)
            try:
                plan, _, _ = learner.ida_star(test_state, goal_state, learner.heuristic, 60)
                if plan:
                    test_solved += 1
            except:
                continue
        
        test_solved_pct = test_solved
        results[length_inc]['test'].append(test_solved_pct)
        print(f"Train Solved: {train_solved_pct:.1f}% | Test Solved: {test_solved_pct}%")
    
    # Print final results
    print("\n=== Final Results ===")
    print("LengthInc\tTrain%\tTest%")
    for li in length_incs:
        train = np.mean(results[li]['train']) if results[li]['train'] else 0
        test = np.mean(results[li]['test']) if results[li]['test'] else 0
        print(f"{li}\t\t{train:.1f}\t{test:.1f}")

# Helper function needed
def generate_task_fixed_steps(goal_state, steps_back):
    current_state = goal_state[:]
    for _ in range(steps_back):
        valid_moves = get_valid_moves(current_state)
        if not valid_moves:
            break
        current_state = random.choice(valid_moves)
    return current_state
    

# Run the experiment
run_lengthinc_experiment()