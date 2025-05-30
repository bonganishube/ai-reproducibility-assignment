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