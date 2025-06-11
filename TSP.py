import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time

def generate_cities(num_cities, seed=42):
    """Tạo ra một tập hợp các thành phố với tọa độ ngẫu nhiên."""
    np.random.seed(seed)
    return np.random.rand(num_cities, 2) * 100


@njit
def calculate_distance_matrix(cities):
    """Tính ma trận khoảng cách Euclid giữa các thành phố."""
    num_cities = cities.shape[0]
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i, num_cities):
            dist = np.sqrt((cities[i, 0] - cities[j, 0]) ** 2 + (cities[i, 1] - cities[j, 1]) ** 2)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix


@njit
def total_distance(path, dist_matrix):
    """Tính tổng quãng đường của một lộ trình."""
    dist = 0.0
    for i in range(len(path) - 1):
        dist += dist_matrix[path[i], path[i + 1]]
    dist += dist_matrix[path[-1], path[0]]  # Quay về thành phố bắt đầu
    return dist


@njit
def swap_mutation(path):
    """Thực hiện đột biến bằng cách hoán đổi vị trí hai thành phố ngẫu nhiên."""
    path_copy = path.copy()
    a, b = np.random.randint(0, len(path_copy), 2)
    while a == b:
        b = np.random.randint(0, len(path_copy))
    path_copy[a], path_copy[b] = path_copy[b], path_copy[a]
    return path_copy


@njit
def inversion_mutation(path):
    """Thực hiện đột biến bằng cách đảo ngược một đoạn trong lộ trình."""
    path_copy = path.copy()
    a, b = np.random.randint(0, len(path_copy), 2)
    if a > b:
        a, b = b, a
    path_copy[a:b + 1] = path_copy[a:b + 1][::-1]
    return path_copy


# ==============================================================================
# ---------- THUẬT TOÁN ELEPHANT HERDING OPTIMIZATION (EHO) ----------
# ==============================================================================
def eho_tsp(cities, n_elephants=50, n_clans=5, n_generations=100, alpha=0.5, beta=0.1):
    num_cities = len(cities)
    dist_matrix = calculate_distance_matrix(cities)
    n_seperate = int(n_elephants / n_clans)

    population = [np.random.permutation(num_cities) for _ in range(n_elephants)]
    fitness = np.array([total_distance(p, dist_matrix) for p in population])

    best_path_global = population[np.argmin(fitness)]
    best_dist_global = np.min(fitness)
    history = [best_dist_global]

    for gen in range(n_generations):
        clans = np.array_split(np.argsort(fitness), n_clans)

        for clan_indices in clans:
            matriarch_idx = clan_indices[0]
            matriarch_path = population[matriarch_idx]

            for elephant_idx in clan_indices[1:]:
                current_path = population[elephant_idx]
                start, end = np.sort(np.random.randint(0, num_cities, 2))

                new_path = np.full(num_cities, -1, dtype=np.int32)
                new_path[start:end] = matriarch_path[start:end]

                fill_pos = list(range(0, start)) + list(range(end, num_cities))
                fill_cities = [city for city in current_path if city not in new_path]

                for i, city in zip(fill_pos, fill_cities):
                    new_path[i] = city

                if np.random.rand() < 0.1:
                    new_path = inversion_mutation(new_path)

                new_dist = total_distance(new_path, dist_matrix)
                if new_dist < fitness[elephant_idx]:
                    population[elephant_idx] = new_path
                    fitness[elephant_idx] = new_dist

        worst_indices = [clan[-1] for clan in clans]
        for idx in worst_indices[:n_seperate]:
            population[idx] = np.random.permutation(num_cities)
            fitness[idx] = total_distance(population[idx], dist_matrix)

        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_dist_global:
            best_dist_global = fitness[current_best_idx]
            best_path_global = population[current_best_idx]

        history.append(best_dist_global)

    return best_path_global, best_dist_global, history


# ==============================================================================
# ---------- THUẬT TOÁN WATER CYCLE ALGORITHM (WCA) ----------
# ==============================================================================
def wca_tsp(cities, n_pop=50, n_rivers=4, n_generations=100, d_max=1e-5):
    num_cities = len(cities)
    dist_matrix = calculate_distance_matrix(cities)

    population = [np.random.permutation(num_cities) for _ in range(n_pop)]
    fitness = np.array([total_distance(p, dist_matrix) for p in population])

    sorted_indices = np.argsort(fitness)
    population = [population[i] for i in sorted_indices]
    fitness = fitness[sorted_indices]

    sea = population[0]
    rivers = population[1:n_rivers + 1]
    streams = population[n_rivers + 1:]

    best_path_global = sea
    best_dist_global = fitness[0]
    history = [best_dist_global]

    for gen in range(n_generations):
        for i in range(len(rivers)):
            new_river = inversion_mutation(rivers[i])
            if np.random.rand() < 0.5:
                start, end = np.sort(np.random.randint(0, num_cities, 2))
                child = np.full(num_cities, -1, dtype=np.int32)
                child[start:end] = sea[start:end]
                fill_cities = [city for city in rivers[i] if city not in child]
                fill_pos = list(range(0, start)) + list(range(end, num_cities))
                for idx_pos, city in zip(fill_pos, fill_cities):
                    child[idx_pos] = city
                new_river = child

            if total_distance(new_river, dist_matrix) < total_distance(rivers[i], dist_matrix):
                rivers[i] = new_river

        for i in range(len(streams)):
            river_to_follow = rivers[i % n_rivers]
            new_stream = inversion_mutation(streams[i])
            if total_distance(new_stream, dist_matrix) < total_distance(streams[i], dist_matrix):
                streams[i] = new_stream

        population = [sea] + rivers + streams
        fitness = np.array([total_distance(p, dist_matrix) for p in population])
        sorted_indices = np.argsort(fitness)
        population = [population[i] for i in sorted_indices]
        fitness = fitness[sorted_indices]

        sea = population[0]
        rivers = population[1:n_rivers + 1]
        streams = population[n_rivers + 1:]

        if total_distance(sea, dist_matrix) - total_distance(rivers[0], dist_matrix) < d_max:
            for i in range(len(streams)):
                streams[i] = inversion_mutation(sea)

        if fitness[0] < best_dist_global:
            best_path_global = sea
            best_dist_global = fitness[0]

        history.append(best_dist_global)

    return best_path_global, best_dist_global, history


# ==============================================================================
# ---------- THUẬT TOÁN ANT COLONY OPTIMIZATION (ACO) ----------
# ==============================================================================
def aco_tsp(cities, n_ants=20, generations=100, alpha=1.0, beta=5.0, evaporation=0.5, Q=100):
    num_cities = len(cities)
    dist_matrix = calculate_distance_matrix(cities)
    pheromone = np.ones((num_cities, num_cities))
    visibility = 1 / (dist_matrix + 1e-10)
    np.fill_diagonal(visibility, 0)

    best_path_global = None
    best_dist_global = float('inf')
    history = []

    for gen in range(generations):
        all_paths = []
        all_dists = []

        for ant in range(n_ants):
            current_city = np.random.randint(0, num_cities)
            path = [current_city]
            unvisited = set(range(num_cities))
            unvisited.remove(current_city)

            while unvisited:
                probs = []
                for next_city in unvisited:
                    tau = pheromone[current_city, next_city] ** alpha
                    eta = visibility[current_city, next_city] ** beta
                    probs.append(tau * eta)

                probs = np.array(probs)
                if probs.sum() == 0:
                    next_city = list(unvisited)[np.random.randint(0, len(unvisited))]
                else:
                    probs /= probs.sum()
                    next_city = np.random.choice(list(unvisited), p=probs)

                path.append(next_city)
                unvisited.remove(next_city)
                current_city = next_city

            dist = total_distance(np.array(path), dist_matrix)
            all_paths.append(path)
            all_dists.append(dist)

            if dist < best_dist_global:
                best_path_global = np.array(path)
                best_dist_global = dist

        pheromone *= (1 - evaporation)
        for path, dist in zip(all_paths, all_dists):
            for i in range(num_cities - 1):
                pheromone[path[i], path[i + 1]] += Q / dist
            pheromone[path[-1], path[0]] += Q / dist

        history.append(best_dist_global)

    return best_path_global, best_dist_global, history


# ==============================================================================
# ---------- TRỰC QUAN HÓA ----------
# ==============================================================================
def plot_tsp_path(cities, path, title='TSP Path'):
    plt.figure(figsize=(8, 7))
    path_coords = cities[path]
    path_coords = np.vstack([path_coords, path_coords[0]])

    plt.plot(path_coords[:, 0], path_coords[:, 1], 'o-', label='Lộ trình', color='dodgerblue', markersize=8,
             markerfacecolor='white')
    plt.plot(cities[:, 0], cities[:, 1], 'o', color='red', markersize=10, label='Thành phố')

    for i, (x, y) in enumerate(cities):
        plt.text(x + 0.5, y + 0.5, str(i), fontsize=10, ha='center', va='center', fontweight='bold', color='black')

    plt.title(title, fontsize=16)
    plt.xlabel("Tọa độ X", fontsize=12)
    plt.ylabel("Tọa độ Y", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_convergence(histories, labels):
    plt.figure(figsize=(12, 6))
    for hist_runs, label in zip(histories, labels):
        mean_hist = np.mean(hist_runs, axis=0)
        std_hist = np.std(hist_runs, axis=0)

        plt.plot(mean_hist, label=label, linewidth=2)
        plt.fill_between(range(len(mean_hist)), mean_hist - std_hist, mean_hist + std_hist, alpha=0.2)

    plt.title("So sánh sự hội tụ trung bình của các thuật toán", fontsize=16)
    plt.xlabel("Thế hệ", fontsize=12)
    plt.ylabel("Tổng quãng đường (Trung bình)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.xlim(0)
    plt.show()


# ==============================================================================
# ---------- CHƯƠNG TRÌNH CHÍNH ----------
# ==============================================================================
if __name__ == "__main__":
    # --- Lấy đầu vào từ người dùng cho số lượng thành phố ---
    while True:
        try:
            num_cities_input = int(input("Nhập số lượng thành phố bạn muốn tạo (>= 3): "))
            if num_cities_input >= 3:
                break
            else:
                print("Lỗi: Số thành phố phải lớn hơn hoặc bằng 3. Vui lòng thử lại.")
        except ValueError:
            print("Lỗi: Đầu vào không hợp lệ. Vui lòng nhập một số nguyên.")

    # --- Cài đặt các tham số khác ---
    NUM_CITIES = num_cities_input
    NUM_RUNS = 5
    NUM_GENERATIONS = 500

    print(f"\nBắt đầu so sánh các thuật toán cho {NUM_CITIES} thành phố.")
    print(f"Mỗi thuật toán sẽ được chạy {NUM_RUNS} lần.\n")

    cities = generate_cities(NUM_CITIES, seed=42)

    algorithms = {
        "EHO": (eho_tsp, {'n_elephants': 50, 'n_clans': 5, 'n_generations': NUM_GENERATIONS}),
        "WCA": (wca_tsp, {'n_pop': 50, 'n_rivers': 4, 'n_generations': NUM_GENERATIONS}),
        "ACO": (aco_tsp, {'n_ants': 30, 'generations': NUM_GENERATIONS, 'alpha': 1.0, 'beta': 3.0})
    }

    results = {}

    for name, (func, params) in algorithms.items():
        print(f"--- Đang chạy {name} ---")
        all_dists = []
        all_paths = []
        all_histories = []
        start_time = time.time()

        for i in range(NUM_RUNS):
            path, dist, history = func(cities, **params)
            all_dists.append(dist)
            all_paths.append(path)
            all_histories.append(history)
            print(f"  Run {i + 1}/{NUM_RUNS}: Quãng đường = {dist:.2f}")

        end_time = time.time()

        results[name] = {
            'dists': np.array(all_dists),
            'paths': all_paths,
            'histories': np.array(all_histories),
            'best_run_idx': np.argmin(all_dists),
            'time': end_time - start_time
        }

    print("\n" + "=" * 80)
    print("KẾT QUẢ THỐNG KÊ SAU " + str(NUM_RUNS) + " LẦN CHẠY")
    print("=" * 80)
    print(
        f"{'Thuật toán':<12} | {'Tốt nhất':<12} | {'Tệ nhất':<12} | {'Trung bình':<12} | {'Độ lệch chuẩn':<15} | {'Thời gian (s)':<15}")
    print("-" * 80)

    for name, data in results.items():
        dists = data['dists']
        print(
            f"{name:<12} | {np.min(dists):<12.2f} | {np.max(dists):<12.2f} | {np.mean(dists):<12.2f} | {np.std(dists):<15.2f} | {data['time']:<15.2f}")

    print("=" * 80 + "\n")

    for name, data in results.items():
        best_idx = data['best_run_idx']
        best_path = data['paths'][best_idx]
        best_dist = data['dists'][best_idx]
        plot_tsp_path(cities, best_path, title=f"Lộ trình tốt nhất của {name} (Quãng đường: {best_dist:.2f})")

    convergence_histories = [results[name]['histories'] for name in algorithms.keys()]
    labels = list(algorithms.keys())
    plot_convergence(convergence_histories, labels)
