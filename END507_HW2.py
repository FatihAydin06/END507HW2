import numpy as np
import pandas as pd
import random

dist1 = pd.read_csv('dist.csv')
flow1 = pd.read_csv('flow.csv')

dist = dist1.to_numpy()
flow = flow1.to_numpy()

iteration = 200
lst_len = 10
solution_len = 20
N = 190
s = 0

def assignment_cost(solution):
  cost = 0
  for i in range(solution_len):
    for j in range(solution_len):
        cost += dist[i][j] *flow[solution[i]][solution[j]]
  return cost

neighbors = np.zeros((N, solution_len + 2), dtype=int)

def swap(sol_n):
    global idx, neighbors
    for i in range (solution_len):
        j = i + 1
        for j in range(solution_len):
            if  i < j:
                idx = idx + 1
                sol_n[j], sol_n[i] = sol_n[i], sol_n[j]
                neighbors[idx, :-2] = sol_n
                neighbors[idx, -2:] = [sol_n[i], sol_n[j]]
                sol_n[i], sol_n[j] = sol_n[j], sol_n[i]

def not_in_tabu (solution, tabu):
    not_found = False
    if not solution.tolist() in tabu:
        solution[0], solution[1] = solution[1], solution[0]
        if not solution.tolist() in tabu:
            not_found = True

    return not_found

def tabu_search():
    global neighbors, iteration, idx, s
    current_solution = [3,11,13,10,2,7,17,19,16,6,5,15,18,14,4,1,10,12,8,0]
    #current_solution = random.sample(range(solution_len), solution_len)
    best_solution = current_solution
    Tabu = []
    frequency = {}

    print("Initial solution: %s Cost: %s " % (current_solution, assignment_cost(current_solution)))
    while s < iteration:

        idx = -1
        swap(current_solution)

        cost = np.zeros((len(neighbors)))
        for index in range(len(neighbors)):
            cost[index] = assignment_cost(neighbors[index, :-2])
        rank = np.argsort(cost)
        neighbors = neighbors[rank]

        for j in range(N):

            not_tabu = not_in_tabu(neighbors[j, -2:], Tabu)
            if (not_tabu):
                current_solution = neighbors[j, :-2].tolist()
                Tabu.append(neighbors[j, -2:].tolist())

                if len(Tabu) > lst_len - 1:
                    Tabu = Tabu[1:]

                if not tuple(current_solution) in frequency.keys():
                    frequency[tuple(current_solution)] = 1

                    if assignment_cost(current_solution) <  assignment_cost(best_solution):
                        best_solution = current_solution
                       
                else:
                    current_cost= assignment_cost(current_solution) + frequency[tuple(current_solution)]
                    frequency[tuple(current_solution)] += 1

                    if current_cost <  assignment_cost(best_solution):
                        best_solution = current_solution
                break

        s += 1
        print("Iteration: %s Solution: %s Cost: %s " % (s, best_solution, assignment_cost(best_solution)))
        print("Tabu List %s" % (Tabu))
    print("Best solution: %s Cost: %s " % (best_solution, assignment_cost(best_solution)))

tabu_search()
