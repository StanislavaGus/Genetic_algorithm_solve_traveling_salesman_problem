from data import Matrix, coorinates, best_path
import numpy as np
import random
import matplotlib.pyplot as plt
import time  # Добавляем для замера времени выполнения

class Gen_alg_Traveling_salesman:
    def __init__(self, distance_matrix, coordinates, best_path):
        self.distance_matrix = distance_matrix
        self.coordinates = coordinates
        self.best_path = best_path
        self.population = None  # Популяция будет инициализироваться в run()

    def create_initial_population(self, population_size):
        """Создает начальную популяцию маршрутов."""
        population = []
        num_cities = len(self.distance_matrix)
        for _ in range(population_size):
            individual = list(range(1, num_cities + 1))
            random.shuffle(individual)
            population.append(individual)
        return population

    def fitness(self, path):
        """Вычисляет приспособленность (общую длину пути)."""
        total_distance = 0
        for i in range(len(path)):
            total_distance += self.distance_matrix[path[i - 1] - 1][path[i] - 1]
        return 1 / total_distance  # Чем меньше расстояние, тем больше приспособленность

    def selection(self):
        """Выбор родителей на основе их приспособленности."""
        fitness_values = [self.fitness(individual) for individual in self.population]
        total_fitness = sum(fitness_values)
        probabilities = [f / total_fitness for f in fitness_values]
        parents = random.choices(self.population, weights=probabilities, k=2)
        return parents

    def crossover_alternating_edges(self, parent1, parent2):
        """кроссинговер с использованием альтернативных ребер."""
        child = [-1] * len(parent1)
        used_cities = set()

        for i in range(len(parent1)):
            if i % 2 == 0:
                for city in parent1:
                    if city not in used_cities:
                        child[i] = city
                        used_cities.add(city)
                        break
            else:
                for city in parent2:
                    if city not in used_cities:
                        child[i] = city
                        used_cities.add(city)
                        break
        return child

    def crossover_subtour_chunks(self, parent1, parent2):
        """кроссинговер с использованием подтуров."""
        child = [-1] * len(parent1)
        used_cities = set()

        chunks = random.sample(range(len(parent1)), 2)
        start, end = sorted(chunks)

        # Копируем подтур от одного родителя
        for i in range(start, end):
            child[i] = parent1[i]
            used_cities.add(parent1[i])

        # Заполняем оставшиеся города из второго родителя
        for i in range(len(parent2)):
            if parent2[i] not in used_cities:
                for j in range(len(child)):
                    if child[j] == -1:
                        child[j] = parent2[i]
                        used_cities.add(parent2[i])
                        break
        return child

    def crossover_heuristic(self, parent1, parent2):
        """Эвристический кроссинговер."""
        child = [-1] * len(parent1)
        start_city = random.choice(parent1)
        child[0] = start_city
        used_cities = {start_city}

        for i in range(1, len(child)):
            last_city = child[i - 1]
            next_city_p1 = parent1[(parent1.index(last_city) + 1) % len(parent1)]
            next_city_p2 = parent2[(parent2.index(last_city) + 1) % len(parent2)]

            # Выбираем ближайший город
            if self.distance_matrix[last_city - 1][next_city_p1 - 1] < self.distance_matrix[last_city - 1][next_city_p2 - 1]:
                next_city = next_city_p1
            else:
                next_city = next_city_p2

            if next_city not in used_cities:
                child[i] = next_city
                used_cities.add(next_city)
            else:
                for city in parent1:
                    if city not in used_cities:
                        child[i] = city
                        used_cities.add(city)
                        break

        return child

    def mutate(self, individual, mutation_rate):
        """Мутация: случайная перестановка двух городов."""
        for swapped in range(len(individual)):
            if random.random() < mutation_rate:
                swap_with = int(random.random() * len(individual))
                individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]

    def run(self, population_size=100, generations=100, mutation_rate=0.01, elitism_rate=0.5, crossover_probability=0.7, crossover_type=0, show_plot=False):
        """Запуск генетического алгоритма с настраиваемыми параметрами и возможностью выводить график."""

        # Начало замера времени
        start_time = time.time()

        # Инициализация популяции
        self.population = self.create_initial_population(population_size)

        best_individual = None
        best_fitness = float('-inf')

        for generation in range(generations):
            new_population = []

            # Сохранение элиты
            num_elites = int(elitism_rate * population_size)
            elites = sorted(self.population, key=self.fitness, reverse=True)[:num_elites]
            new_population.extend(elites)

            # Заполнение новой популяции
            while len(new_population) < population_size:
                parent1, parent2 = self.selection()

                # Решаем, происходит ли кроссинговер
                if random.random() < crossover_probability:
                    # Выбираем тип кроссинговера
                    if crossover_type == 0:
                        child = self.crossover_alternating_edges(parent1, parent2)
                    elif crossover_type == 1:
                        child = self.crossover_subtour_chunks(parent1, parent2)
                    elif crossover_type == 2:
                        child = self.crossover_heuristic(parent1, parent2)
                    else:
                        raise ValueError("Некорректный тип кроссинговера. Используйте 0, 1 или 2.")
                else:
                    if random.random() <= 0.5:
                        child = parent1.copy()
                    else:
                        child = parent2.copy()

                self.mutate(child, mutation_rate)
                new_population.append(child)

            self.population = new_population

            # Обновление лучшего маршрута
            for individual in self.population:
                current_fitness = self.fitness(individual)
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_individual = individual

        # Вычисляем длину найденного маршрута
        best_distance = self.calculate_distance(best_individual)
        # Вычисляем длину оптимального маршрута
        optimal_distance = self.calculate_distance(self.best_path)

        # Конец замера времени
        execution_time = time.time() - start_time

        # Результаты
        print(f'Тип кроссинговера: {crossover_type}')
        print(f'Найденный лучший путь: {best_individual}')
        print(f'Длина найденного пути: {best_distance}')
        print(f'Оптимальный путь: {self.best_path}')
        print(f'Длина оптимального пути: {optimal_distance}')
        print(f'Время выполнения: {execution_time:.4f} секунд\n')

        # Построение графика с указанием типа кроссинговера, если флаг show_plot=True
        if show_plot:
            self.plot_solution(best_individual, crossover_type)

    def calculate_distance(self, path):
        """Вычисляет длину маршрута."""
        total_distance = 0
        for i in range(len(path)):
            total_distance += self.distance_matrix[path[i] - 1][path[(i + 1) % len(path)] - 1]
        return total_distance

    def plot_solution(self, best_path, crossover_type):
        """Визуализация найденного маршрута с указанием типа кроссинговера."""
        complete_path = best_path + [best_path[0]]
        path_coordinates = [self.coordinates[i - 1] for i in complete_path]

        x, y = zip(*path_coordinates)

        # Определяем название типа кроссинговера
        crossover_names = {
            0: "Alternating Edges Crossover",
            1: "Subtour Chunks Crossover",
            2: "Heuristic Crossover"
        }
        crossover_name = crossover_names.get(crossover_type, "Unknown Crossover")

        plt.figure(figsize=(10, 8))
        plt.plot(x, y, marker='o', linestyle='-')
        plt.title(f'Найденный маршрут коммивояжера\nТип кроссинговера: {crossover_name}')
        plt.xlabel('Координаты X')
        plt.ylabel('Координаты Y')
        plt.grid()
        plt.scatter(*zip(*self.coordinates), c='red', s=50)
        plt.show()


# Создание объекта класса
data_matrix = Matrix()
genetic_algorithm = Gen_alg_Traveling_salesman(data_matrix.distance_matrix, coorinates, best_path)

# Параметры для экспериментов
population_size = 150
generations = 150

# Настройки для экспериментов: низкая/высокая мутация и низкий/высокий кроссинговер
crossover_probabilities = [0.25, 0.5, 0.75]  # Низкая и высокая вероятность кроссинговера
mutation_rates = [0.001, 0.01, 0.05]  # Низкая и высокая вероятность мутации

# Переменные для экспериментов
crossover_types = {
    0: "Alternating Edges Crossover",
    1: "Subtour Chunks Crossover",
    2: "Heuristic Crossover"
}

# Функция для проведения серии экспериментов
def run_experiment(crossover_type, mutation_rate, crossover_probability, show_plot_flag = False):
    print(f"\nЭксперимент с типом кроссинговера: {crossover_types[crossover_type]}")
    print(f"Параметры: Вероятность мутации = {mutation_rate}, Вероятность кроссинговера = {crossover_probability}")
    genetic_algorithm.run(
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_probability=crossover_probability,
        crossover_type=crossover_type,
        show_plot=show_plot_flag
    )

#run_experiment(0, 0.001, 0.75, True)
#run_experiment(1, 0.001, 0.75, True)
run_experiment(2, 0.001, 0.75, True)

experement_flag = True

if experement_flag:
    # Цикл по типам кроссинговера и параметрам мутации/кроссинговера
    for crossover_type in range(1,3):
        for mutation_rate in mutation_rates:
            for crossover_probability in crossover_probabilities:
                run_experiment(crossover_type, mutation_rate, crossover_probability)
