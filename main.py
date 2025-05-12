import random
import time


def path_distance(path, matrix): #use the matrix and path array to calculate the distance
    dist = 0
    for i in range(len(path)):
        dist += matrix[path[i]][path[(i + 1) % len(path)]]
    return dist


# ---------------------- Genetic Algorithm ----------------------
def ga_tsp(matrix, num_generations=500, population_size=100, mutation_rate=0.1):
    num_cities = len(matrix)

    #make a random path (chromosome)
    def create_individual():
        individual = list(range(num_cities))
        random.shuffle(individual)
        return individual

    # we can't randomly change a value, cuz we don't want cities to repeat
    def mutate(individual):
        i, j = random.sample(range(num_cities), 2)
        individual[i], individual[j] = individual[j], individual[i]

    def crossover(parent1, parent2):
        start, end = sorted(random.sample(range(num_cities), 2)) #select 2 points[-1] * num_cities
        child = [-1] * num_cities
        child[start:end] = parent1[start:end]
        ptr = 0
        for city in parent2:   #iterate parent2 and add the new cities
            if city not in child:
                while child[ptr] != -1:   #find the first empty slot
                    ptr += 1
                child[ptr] = city
        return child

    population = [create_individual() for _ in range(population_size)]

    for generation in range(num_generations):
        population.sort(key=lambda x: path_distance(x, matrix)) #sort by dist
        next_generation = population[:10]  # Elitism
        while len(next_generation) < population_size:
            parent1, parent2 = random.choices(population[:50], k=2)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:   #random number from 0 to 1
                mutate(child)
            next_generation.append(child)
        population = next_generation

    best = min(population, key=lambda x: path_distance(x, matrix))
    return best, path_distance(best, matrix)

# ---------------------- Ant Colony Optimization ----------------------
def aco_tsp(matrix, num_ants=20, num_iterations=100, alpha=1, beta=5, evaporation=0.5, Q=100):
    num_cities = len(matrix)
    pheromone = [[1 for _ in range(num_cities)] for _ in range(num_cities)] #10by10 pheromone matrix

    def select_next_city(visited, current):
        probs = []
        for j in range(num_cities):
            if j in visited or matrix[current][j] == float('inf'):
                probs.append(0)
            else:
                probs.append((pheromone[current][j] ** alpha) * ((1 / matrix[current][j]) ** beta))
        total = sum(probs)
        if total == 0:
            return random.choice([j for j in range(num_cities) if j not in visited])
        probs = [p / total for p in probs]
        r = random.random()
        cum = 0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                return i

    best_path = None
    best_dist = float('inf')

    for iteration in range(num_iterations):
        all_paths = []
        for _ in range(num_ants):
            path = [random.randint(0, num_cities - 1)]
            while len(path) < num_cities:
                next_city = select_next_city(path, path[-1])
                path.append(next_city)
            dist = path_distance(path, matrix)
            if dist < best_dist:
                best_path = path[:]
                best_dist = dist
            all_paths.append((path, dist))

        for i in range(num_cities):
            for j in range(num_cities):
                pheromone[i][j] *= (1 - evaporation)

        for path, dist in all_paths:
            for i in range(len(path)):
                a = path[i]
                b = path[(i + 1) % num_cities]
                pheromone[a][b] += Q / dist
                pheromone[b][a] += Q / dist

    return best_path, best_dist


def main():
    #read file
    with open('in.txt', 'r') as f:
        lines = f.readlines()
    n = int(lines[0].strip())   #first line: number of cities
    matrix = []
    for line in lines[1:]:
        row = list(map(int, line.strip().split())) #every number in the line/row
        matrix.append([float('inf') if x == -1 else x for x in row]) #inf for when a path doesn't exist

    start_ga = time.time()
    ga_path, ga_dist = ga_tsp(matrix)
    end_ga = time.time()

    start_aco = time.time()
    aco_path, aco_dist = aco_tsp(matrix)
    end_aco = time.time()

    print("Genetic Algorithm Result:")
    print("Path:", ga_path)
    print("Distance:", ga_dist)
    print("Time:", end_ga - start_ga, "seconds\n")

    print("Ant Colony Optimization Result:")
    print("Path:", aco_path)
    print("Distance:", aco_dist)
    print("Time:", end_aco - start_aco, "seconds")

if __name__ == '__main__':
    main()
