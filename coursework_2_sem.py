import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple
from tabulate import tabulate


def runge_kutta_3(f: Callable, y0: np.ndarray, t_span: Tuple[float, float],
                  h: float) -> Tuple[np.ndarray, np.ndarray]:
    t0, tf = t_span
    n_steps = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, n_steps)

    n = len(y0)
    y = np.zeros((n_steps, n))
    y[0] = y0

    for i in range(n_steps - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 3, y[i] + h / 3 * k1)
        k3 = f(t[i] + 2 * h / 3, y[i] + 2 * h / 3 * k2)

        y[i + 1] = y[i] + h * (k1 + 3 * k3) / 4

    return t, y


def ttest_system(t: float, y: np.ndarray) -> np.ndarray:
    y1, y2 = y
    dy1 = -np.sin(t) / np.sqrt(1 + np.exp(2 * t)) + y1 * (y1 ** 2 + y2 ** 2 - 1)
    dy2 = np.cos(t) / np.sqrt(1 + np.exp(2 * t)) + y2 * (y1 ** 2 + y2 ** 2 - 1)
    return np.array([dy1, dy2])


def ttest_exact_solution(t: np.ndarray) -> np.ndarray:
    y1 = np.cos(t) / np.sqrt(1 + np.exp(2 * t))
    y2 = np.sin(t) / np.sqrt(1 + np.exp(2 * t))
    return np.column_stack((y1, y2))


def predator_prey_system(t: float, y: np.ndarray, params: dict) -> np.ndarray:
    X, Y = y
    alpha = params['alpha']
    epsilon = params['epsilon']
    gamma = params['gamma']

    dX = (1 - epsilon * X) * X - (X * Y) / (1 + alpha * X)
    dY = gamma * ((X / (1 + alpha * X)) - 1) * Y

    return np.array([dX, dY])


def find_equilibrium_points(params: dict) -> List[np.ndarray]:
    alpha = params['alpha']
    epsilon = params['epsilon']

    eq1 = np.array([0, 0])  # Тривиальное равновесие (0, 0)
    eq2 = np.array([1 / epsilon, 0])  # Равновесие без хищников (1/ε, 0)

    # Нетривиальное равновесие
    X_eq = 1 / (1 - alpha)
    Y_eq = (1 - epsilon * X_eq) * (1 + alpha * X_eq)

    equilibria = [eq1, eq2]
    
    equilibria.append(np.array([X_eq, abs(Y_eq)]))

    return equilibria


def calculate_error(numerical: np.ndarray, exact: np.ndarray) -> float:
    return np.max(np.abs(numerical - exact))


def main():
    t_span = (0, 5)
    y0_test = np.array([1 / np.sqrt(2), 0])

    h_values = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625]
    errors = []

    plt.figure(figsize=(12, 8))

    for h in h_values:
        t, y_numerical = runge_kutta_3(ttest_system, y0_test, t_span, h)
        y_exact = ttest_exact_solution(t)

        error = calculate_error(y_numerical, y_exact)
        errors.append(error)

    print("\nАнализ порядка точности метода Рунге-Кутты 3-го порядка")
    print("=" * 60)

    # Вывод таблицы погрешностей
    print("\nТаблица погрешностей:")
    print(f"{'Шаг h':<12} | {'Погрешность e':<15}")
    print("-" * 30)
    for i, h in enumerate(h_values):
        print(f"{h:<12.8f} | {errors[i]:<15.10e}")

    print("\nВычисление порядка точности:")
    print(f"{'Шаг h':<12} | {'Порядок α':<10}")
    print("-" * 25)

    alphas = []
    for i in range(len(h_values) - 1):
        h_i = h_values[i]
        h_i_plus_1 = h_values[i + 1]
        e_i = errors[i]
        e_i_plus_1 = errors[i + 1]

        # Вычисляем порядок точности
        alpha = np.log10(e_i_plus_1 / e_i) / np.log10(h_i_plus_1 / h_i)
        alphas.append(alpha)

        print(f"{h_i:<12.8f} | {alpha:<10.4f}")

    # Вычисляем среднее значение порядка точности
    avg_alpha = sum(alphas) / len(alphas)
    print("-" * 25)
    print(f"Среднее значение α = {avg_alpha:.4f}")



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.loglog(h_values, errors, 'o-', label='Погрешность')
    plt.loglog(h_values, [h ** 3 for h in h_values], '--', label='h³')
    plt.xlabel('Шаг h')
    plt.ylabel('Максимальная погрешность')
    plt.title('Зависимость погрешности от шага')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(h_values, [error / (h ** 3) for error, h in zip(errors, h_values)], 'o-')
    plt.xlabel('Шаг h')
    plt.ylabel('Погрешность / h³')
    plt.title('Отношение погрешности к h³')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('error_analysis.png')

    alpha_values = np.linspace(0.1, 0.9, 9)
    table_data = []
    t_span_pp = (0, 50)
    y0_pp = np.array([3, 1])  # Начальные условия для системы хищник-жертва
    

    for alpha in alpha_values:
        params = {'alpha': alpha, 'epsilon': 0.1, 'gamma': 1}
        equilibria = find_equilibrium_points(params)

        row = [f"{alpha:.1f}"]
        for eq in equilibria:
            row.extend([f"{eq[0]:.4f}", f"{eq[1]:.4f}"])
        table_data.append(row)
    
    headers = ["α", "X1", "Y1", "X2", "Y2", "X3", "Y3"]
    for row in table_data:
        while len(row) < len(headers):
            row.append("")

    print("\nСтационарные точки системы хищник-жертва:")
    print ("epsilon = 0.1, gamma = 1")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    plt.figure(figsize=(15, 10))

    for i, alpha in enumerate(alpha_values):
        params = {'alpha': alpha, 'epsilon': 0.1, 'gamma': 1}

        def system_with_params(t, y):
            return predator_prey_system(t, y, params)

        t, y = runge_kutta_3(system_with_params, y0_pp, t_span_pp, 0.01)

        # Построение фазового портрета
        plt.subplot(3, 3, i + 1)
        plt.plot(y[:, 0], y[:, 1])
        plt.scatter(y0_pp[0], y0_pp[1], color='red', label='Начальная точка')

        # Отмечаем стационарные точки
        equilibria = find_equilibrium_points(params)
        for eq in equilibria:
            if eq[0] > 0 and eq[1] > 0:
                plt.scatter(eq[0], eq[1], color='green', marker='x')

        plt.xlabel('X (жертвы)')
        plt.ylabel('Y (хищники)')
        plt.title(f'α = {alpha:.1f}')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('phase_portraits.png')


    # characteristic_alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]
    characteristic_alphas = [0.1, 0.5, 0.9]

    plt.figure(figsize=(15, 15))

    for i, alpha in enumerate(characteristic_alphas):
        params = {'alpha': alpha, 'epsilon': 0.1, 'gamma': 1}

        def system_with_params(t, y):
            return predator_prey_system(t, y, params)

        t, y = runge_kutta_3(system_with_params, y0_pp, t_span_pp, 0.01)

        # Фазовый портрет
        plt.subplot(3, 3, i * 3 + 1)
        plt.plot(y[:, 0], y[:, 1])
        plt.scatter(y0_pp[0], y0_pp[1], color='red', label='Начальная точка')

        # Отмечаем стационарные точки
        equilibria = find_equilibrium_points(params)
        for eq in equilibria:
            if eq[0] > 0 and eq[1] > 0:
                plt.scatter(eq[0], eq[1], color='green', marker='x')

        plt.xlabel('X (жертвы)')
        plt.ylabel('Y (хищники)')
        plt.title(f'Фазовый портрет, α = {alpha:.1f}')
        plt.grid(True)

        # График X(t)
        plt.subplot(3, 3, i * 3 + 2)
        plt.plot(t, y[:, 0])
        plt.xlabel('Время t')
        plt.ylabel('X (жертвы)')
        plt.title(f'Динамика жертв, α = {alpha:.1f}')
        plt.grid(True)

        # График Y(t)
        plt.subplot(3, 3, i * 3 + 3)
        plt.plot(t, y[:, 1])
        plt.xlabel('Время t')
        plt.ylabel('Y (хищники)')
        plt.title(f'Динамика хищников, α = {alpha:.1f}')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('detailed_analysis.png')


if __name__ == "__main__":
    main()
