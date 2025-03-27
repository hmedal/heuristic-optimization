import pandas as pd
import gurobipy as gp
from gurobipy import GRB, Model
from gurobipy import quicksum as qsum
import time
import networkx as nx
import matplotlib.pyplot as plt
import math
import random
from collections import defaultdict
from heapq import heappush, heappop

def complete_tour(tour : list):
    return tour + [tour[0]]

def get_cost_of_edges(edges_used, costs : dict):
    return sum(costs[u,v] for u, v in edges_used)

def get_cost_of_tour(tour : list, costs):
    edges_used = list(zip(tour, tour[1:]))
    return round(get_cost_of_edges(edges_used, costs))

def get_tour_from_edges_used(edges_used :list):
    tour = []
    next_map = {i : j for (i, j) in edges_used}
    current = edges_used[0][0]
    tour.append(current)
    while len(tour) < len(edges_used) + 1:
        current = next_map[current]
        tour.append(current)
    return tour

def get_edges_used_from_tour(tour : list):
    return [(tour[i], tour[i+1]) for i in range(len(tour) - 1)]

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_time(p1, p2, time_conversion = 1, avg_speed = 1):
    """ 
        Get time in minutes to travel from p1 to p2 
    """
    return time_conversion*(euclidean_distance(p1, p2)/avg_speed)

def get_cost(p1, p2, cost_per_mile):
    return cost_per_mile*euclidean_distance(p1, p2)

def handle_model(m : Model):
    if m.Status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        m.computeIIS()

        print("The following constraints are in the IIS:")
        for c in m.getConstrs():
            if c.IISConstr:
                print(f"  {c.ConstrName}")

        for v in m.getVars():
            if v.IISLB or v.IISUB:
                print(f"  Variable bound conflict: {v.VarName}")

def get_coordinate(locations : dict, node : int):
    return (locations[node]['XCOORD'], locations[node]['YCOORD'])

def get_rand_color():
    return (random.random(), random.random(), random.random())

def show_tours(edges_used_list, locations, to_tikz = False, 
               num_vehicles = 1,
               tikz_file_name = "tikz.tex"):
    colors = [get_rand_color() for k in  range(num_vehicles)]
    G = nx.DiGraph()
    for node in locations:
        G.add_node(node, pos=get_coordinate(locations, node))
    #edge_colors = []
    i = 0
    for edges_used in edges_used_list:
        for edge in edges_used:
            #edge_colors.append(colors[i])
            G.add_edge(edge[0], edge[1], color = colors[i])
        i += 1
    edge_colors = [G[u][v]['color'] for u,v in G.edges()]
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos=pos, with_labels=True, edge_color=edge_colors)
    plt.show()
    #formatted_labels = {k: f"{int(v)}" for k, v in labels.items()}  # Format the edge labels to the nearest integer
    if to_tikz:
        tikz_opts = "scale=0.2, ultra thick, node_style/.style={circle,draw=blue,fill=blue!20!,scale=0.6,font=\\tiny},edge_style/.style={draw=black, thick,font=\\tiny}"
        edge_opts = {e: "edge_style" for e in G.edges}
        edge_label_opts = {e: "below" for e in G.edges}
        node_opts = {i: "node_style" for i in G.nodes}
        latex_code = nx.to_latex(G,
                        tikz_options = tikz_opts,
                        node_options=node_opts, 
                        edge_options=edge_opts, 
                        #edge_label=formatted_labels,
                        edge_label_options=edge_label_opts, 
                        as_document = False, caption="test", 
                        latex_label="fig:soln")
        with open(tikz_file_name, "w") as f:
            f.write(latex_code)

def solve_model(m : Model, show_log = False):
    start_time = time.time()
    m.Params.LogToConsole = show_log
    m.optimize()
    handle_model(m)
    print("Optimal objective value:", round(m.ObjVal))
    end_time = time.time()
    execution_time = round(end_time - start_time, 3)
    print("Optimization time:", execution_time, "seconds")

def create_tsp_model(depot : int, nodes, cost : dict, relaxed = False):
    m = gp.Model("TSP-depot")
    if relaxed:
        var_type = GRB.CONTINUOUS
    else:
        var_type = GRB.BINARY
    x = m.addVars(nodes, nodes, name="x", vtype=var_type)
    u = m.addVars(nodes, name="u")

    m.setObjective(qsum(cost[i, j]*x[i, j] for i in nodes for j in nodes),
                GRB.MINIMIZE)

    m.addConstrs((qsum(x[i,h] for h in nodes if h != i) == 1 for i in nodes), 
                name="Out")

    m.addConstrs((qsum(x[h,i] for h in nodes if h != i) == 1 for i in nodes), 
                name="In")

    m.addConstrs((u[i] - u[j] + len(nodes) * x[i, j] <= len(nodes) - 1 
                for i in nodes for j in nodes if i != depot and j != depot), 
                name="Subtour")
    m._x = x
    m._u = u
    return m

def create_tsp_model_subtour_elim(nodes, cost : dict, relaxed = False):
    m = gp.Model("TSP-subtour-elim")
    if relaxed:
        var_type = GRB.CONTINUOUS
    else:
        var_type = GRB.BINARY
    x = m.addVars(nodes, nodes, name="x", ub = 1, vtype=var_type)

    m.setObjective(qsum(cost[i, j]*x[i, j] for i in nodes for j in nodes),
                GRB.MINIMIZE)

    m.addConstrs((qsum(x[i,h] for h in nodes if h != i) == 1 for i in nodes), 
                name="Out")

    m.addConstrs((qsum(x[h,i] for h in nodes if h != i) == 1 for i in nodes), 
                name="In")
    m._x = x
    return m

def get_minimal_cycles_in_directed_graph(arcs_used):
    node_neighbors = defaultdict(list) # Create a mapping from each node to its neighbours
    for i, j in arcs_used:
        node_neighbors[i].append(j)
    assert all(len(neighbors) == 1 for neighbors in node_neighbors.values())
    
    unvisited = list(node_neighbors) # All nodes are unvisited

    min_heap_cycles = []
   
    def dfs(node, root, path):
        path.append(node)
        for neighbor in node_neighbors[node]:
            if neighbor == root:
                heappush(min_heap_cycles, (-len(path), path))
            elif neighbor in unvisited:
                dfs(neighbor,root, path)
                unvisited.remove(neighbor)
        
    while unvisited: # until unvisited is empty set
        current = unvisited.pop()
        dfs(current, current, [])
    min_len = min_heap_cycles[0][0]
    res = []
    while min_heap_cycles:
        if min_heap_cycles[0][0] == min_len:
            res.append(heappop(min_heap_cycles)[1])
        if min_heap_cycles and min_heap_cycles[0][0] > min_len:
            break
    return res

def round_tsp_solution(edges_used_adjacency_list, n : int, log = True) -> int:
    current = list(edges_used_adjacency_list.keys())[0]
    tour = [current]
    while len(tour) < n:
        feasible_next = {j : edges_used_adjacency_list[current][j] 
                    for j in edges_used_adjacency_list[current] if j not in tour}
        if log:
            filtered = {j : feasible_next[j] 
                        for j in feasible_next if feasible_next[j] > 0.001}
            print(current, filtered)
        next = max(feasible_next, key=feasible_next.get)
        tour.append(next)
        current = next
    return tour

def solve_tsp_tw(depot : int, nodes, locations : dict, travel_time : dict, cost : dict):
    m = gp.Model("TSP-depot-TW")

    x = m.addVars(nodes, nodes, name="x", vtype=GRB.BINARY)
    t = m.addVars(nodes, name="t")
    return_time = m.addVar(name="return_time")

    m.setObjective(qsum(cost[i, j]*x[i, j] for i in nodes for j in nodes),
                GRB.MINIMIZE)

    m.addConstrs((qsum(x[i,h] for h in nodes if h != i) == 1 for i in nodes), 
                name="Out")

    m.addConstrs((qsum(x[h,i] for h in nodes if h != i) == 1 for i in nodes), 
                name="In")

    for i in nodes:
        e_i = locations[i]['READY_TIME']
        l_i = locations[i]['DUE_DATE']
        m.addConstr(t[i] >= e_i, name=f"TW_lower_{i}")
        m.addConstr(t[i] <= l_i, name=f"TW_upper_{i}")

    for i in nodes:
            for j in nodes:
                if i != j and j != depot:
                    service_i = locations[i]['SERVICE_TIME']
                    m.addConstr(
                        t[j] >= t[i] + service_i + travel_time[(i, j)] - M * (1 - x[i, j]),
                        name=f"TimeLink_{i}_{j}"
                )
    m.addConstrs((return_time >= t[i] + service_i + travel_time[(i, j)] - M * (1 - x[i, depot]) for i in nodes), 
                "ReturnTime")
    m.addConstr(t[depot] == 0, "DepotStart")
    m.write("tsp_tw.lp")

    m.Params.LogToConsole = 0 #turn off output
    start_time = time.time()
    m.optimize()
    print("Optimal objective value:", round(m.ObjVal))
    end_time = time.time()
    execution_time = round(end_time - start_time, 3)
    print("Optimization time:", execution_time, "seconds")

    # edges_used_gurobi_tw = [(i, j) for i, j in edges if x[i, j].x > 0.5]
    # tour_tw = get_tour_from_edges_used(edges_used_gurobi_tw)
    # print("edges used", edges_used_gurobi_tw)
    # print("tour", tour_tw)
    # for i in tour_tw[:-1]:
    #     print("Node", i, "visited at time", t[i].x)
    # print("return time", return_time.x)

class VRPInstance():
    
    def __init__(self, data_file, num_customers,
                 num_vehicles, depot, vehicle_capacity,
                 average_speed, cost_per_mile) -> None:
        
        self.read_data_file(data_file, num_customers)
        self.depot = depot
        self.n = len(self.locations)
        self.vehicles = range(num_vehicles)
        self.vehicle_capacity = vehicle_capacity
        self.average_speed = average_speed
        self.cost_per_mile = cost_per_mile
        self.create_sets()
        self.create_dicts()

    def read_data_file(self, data_file : str, num_customers):
        data_df = pd.read_csv(data_file, sep='\s+')
        data_dict_rows = data_df.to_dict(orient='records')
        self.locations = {row['CUST_NO'] : row 
                     for row in data_dict_rows if row['CUST_NO'] <= num_customers}
    
    def create_sets(self):
        self.nodes = list(self.locations.keys())
        self.edges = [(i,j) for i in self.nodes for j in self.nodes if i != j]

    def create_dicts(self):
        self.dist = {}
        self.edge_costs = {}
        self.travel_time = {}
        for i in self.nodes:
            for j in self.nodes:
                x = get_coordinate(self.locations, i)
                y = get_coordinate(self.locations, j)
                self.dist[(i, j)] = euclidean_distance(x, y)
                self.edge_costs[(i, j)] = get_cost(x, y, self.cost_per_mile)
                self.travel_time[(i, j)] = get_time(x, y, 
                                            avg_speed = self.average_speed)

class VRPModel():

    def __init__(self, instance) -> None:
        self.instance = instance
        nodes = self.instance.nodes
        self.node_pairs = [(i,j) for i in nodes for j in nodes]

    def get_gurobipy_model(self, relaxed=False):
        nodes = self.instance.nodes
        vehicles = self.instance.vehicles
        edge_costs = self.instance.edge_costs
        depot = self.instance.depot
        locations = self.instance.locations
        vehicle_capacity = self.instance.vehicle_capacity
        travel_time = self.instance.travel_time
        M = 1000000
        if relaxed:
            var_type = GRB.CONTINUOUS
        else:
            var_type = GRB.BINARY
    
        m = gp.Model("VRP-TW")

        x = m.addVars(nodes, nodes, vehicles, name="x", vtype=var_type)
        t = m.addVars(nodes, vehicles, name="t")
        return_time = m.addVars(vehicles, name="return_time")

        m.setObjective(qsum(edge_costs[i, j]*x[i, j, k] for i in nodes 
                            for j in nodes for k in vehicles),
                    GRB.MINIMIZE)

        m.addConstrs((qsum(x[i,h,k] for h in nodes for k in vehicles if h != i) == 1 
                    for i in nodes if i != depot), 
                    name="Out")

        for k in vehicles:
            for i in nodes:
                if i != depot:
                    m.addConstr(
                        gp.quicksum(x[h, i,k] for h in nodes if h != i) ==
                        gp.quicksum(x[i, j,k] for j in nodes if j != i),
                        name=f"Flow_{k}_{i}"
                    )
        
        for k in vehicles:
            m.addConstr(
                gp.quicksum(locations[i]['DEMAND'] * gp.quicksum(x[i, j, k] 
                                                                for j in nodes if j != i)
                            for i in nodes) <= vehicle_capacity,
                name=f"Capacity_{k}"
            )

        for k in vehicles:
            m.addConstr(
                gp.quicksum(x[depot, j,k] for j in nodes if j != depot) == 1,
                name=f"LeaveDepot_{k}"
            )
            m.addConstr(
                gp.quicksum(x[i, depot,k] for i in nodes if i != depot) == 1,
                name=f"ReturnToDepot_{k}"
            )
        for i in nodes:
            for k in vehicles:
                e_i = locations[i]['READY_TIME']
                l_i = locations[i]['DUE_DATE']
                m.addConstr(t[i,k] >= e_i, name=f"TW_lower_{i}")
                m.addConstr(t[i,k] <= l_i, name=f"TW_upper_{i}")

        for i,j in self.node_pairs:
                if i != j and j != depot:
                    for k in vehicles:
                        service_i = locations[i]['SERVICE_TIME']
                        rhs = t[i,k] + service_i + travel_time[(i, j)] - M * (1 - x[i, j,k])
                        m.addConstr(
                            t[j,k] >= rhs,
                            name=f"TimeLink_{i}_{j}_{k}"
                        )
        for i in nodes:
            rhs = t[i,k] + service_i + travel_time[(i, depot)] - M * (1 - x[i, depot, k])
            m.addConstr(
                return_time[k] >= rhs,
                name="ReturnTime"
            )
        m.addConstr(t[depot,k] == 0, "DepotStart")
        m._x = x
        m._t = t
        m._return_time = return_time
        return m