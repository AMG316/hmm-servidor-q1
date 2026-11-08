import numpy as np
import pandas as pd

# 1. Fijar semilla aleatoria para garantizar reproducibilidad exacta
SEED = 42
np.random.seed(SEED)

# 2. Definición formal del HMM
states = ["Normal", "Degradado", "Falla Crítica"]
obs_labels = ["Latencia Baja", "Latencia Moderada", "Time-Out"]

# Matriz A (Transición)
A_nominal = np.array([
    [0.90, 0.08, 0.02],
    [0.10, 0.75, 0.15],
    [0.05, 0.15, 0.80]
])

# Matriz B (Emisión Nominal - E1)
B_nominal = np.array([
    [0.85, 0.12, 0.03],
    [0.15, 0.70, 0.15],
    [0.02, 0.18, 0.80]
])

# Matriz B (Emisión Perturbada/Ruidosa - E2)
B_perturbada = np.array([
    [0.60, 0.25, 0.15],
    [0.25, 0.50, 0.25],
    [0.15, 0.35, 0.50]
])

# Vector Inicial (pi)
pi = np.array([0.80, 0.15, 0.05])

# 3. Simulación Estocástica
def simulate_hmm(T, A, B, pi):
    N, M = B.shape
    hidden = np.zeros(T, dtype=int)
    obs = np.zeros(T, dtype=int)
    hidden[0] = np.random.choice(N, p=pi)
    obs[0] = np.random.choice(M, p=B[hidden[0]])
    for t in range(1, T):
        hidden[t] = np.random.choice(N, p=A[hidden[t-1]])
        obs[t] = np.random.choice(M, p=B[hidden[t]])
    return hidden, obs

# 4. Algoritmo de Viterbi (Decodificación)
def viterbi(obs, A, B, pi):
    N = A.shape[0]
    T = len(obs)
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    
    delta[0, :] = pi * B[:, obs[0]]
    
    for t in range(1, T):
        for j in range(N):
            temp = delta[t-1, :] * A[:, j]
            delta[t, j] = np.max(temp) * B[j, obs[t]]
            psi[t, j] = np.argmax(temp)
            
    q_star = np.zeros(T, dtype=int)
    q_star[T-1] = np.argmax(delta[T-1, :])
    for t in range(T-2, -1, -1):
        q_star[t] = psi[t+1, q_star[t+1]]
    return q_star

# 5. Experimento Monte Carlo (100 simulaciones por escenario)
def run_experiment(A, B, n_sim=100, T=100):
    accs = []
    for _ in range(n_sim):
        true_s, obs_s = simulate_hmm(T, A, B, pi)
        pred_s = viterbi(obs_s, A, B, pi)
        accs.append(np.mean(true_s == pred_s))
    
    mean_acc = np.mean(accs)
    std_acc = np.std(accs, ddof=1)
    ic_margin = 1.96 * (std_acc / np.sqrt(n_sim))
    return mean_acc, std_acc, np.min(accs), np.max(accs), (mean_acc - ic_margin, mean_acc + ic_margin)

# Ejecución
m1, s1, min1, max1, ic1 = run_experiment(A_nominal, B_nominal)
m2, s2, min2, max2, ic2 = run_experiment(A_nominal, B_perturbada)

print("="*60)
print(" TABLA DE RESULTADOS DE SIMULACIÓN (100 EJECUCIONES)")
print("="*60)
print(f"E1 Nominal    : Media={m1:.3f} | Std={s1:.3f} | Min={min1:.3f} | Max={max1:.3f} | IC 95%=[{ic1[0]:.3f}, {ic1[1]:.3f}]")
print(f"E2 Perturbado : Media={m2:.3f} | Std={s2:.3f} | Min={min2:.3f} | Max={max2:.3f} | IC 95%=[{ic2[0]:.3f}, {ic2[1]:.3f}]")
print("="*60)
