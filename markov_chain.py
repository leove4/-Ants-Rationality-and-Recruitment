import numpy as np
import random
import matplotlib.pyplot as plt


def transition_probabilities(N, eps, delta):
    """
    Construire et retourner la matrice de transition (N+1)x(N+1) P, où P[k, j] = P(état=k -> état=j).
    """
    if eps >= (1 - delta) / (N - 1):
        raise ValueError(f"Condition violée : ε doit être < {(1 - delta)/(N - 1):.6f}. Valeurs fournies : ε={eps}, δ={delta}, N={N}.")
    P = np.zeros((N + 1, N + 1))
    for k in range(N + 1):
        # probabilité d'aller de k à k+1
        p1 = (1 - k / N) * (eps + (1 - delta) * (k / (N - 1))) if k < N else 0.0
        # probabilité d'aller de k à k-1
        p2 = (k / N) * (eps + (1 - delta) * ((N - k) / (N - 1))) if k > 0 else 0.0
        p3 = 1.0 - p1 - p2
        if k < N:
            P[k, k + 1] = p1
        if k > 0:
            P[k, k - 1] = p2
        P[k, k] = p3
    return P

def simulate_chain(N, eps, delta, initial_k=None, num_steps=100000):
    """
    Simule la chaîne de Markov pendant un nombre donné d'itérations. Renvoie la séquence des états visités.
    """
    if initial_k is None:
        initial_k = N // 2
    k = initial_k
    states = [k]
    for _ in range(num_steps):
        # calculer les probabilités de transition
        p1 = (1 - k / N) * (eps + (1 - delta) * (k / (N - 1))) if k < N else 0.0
        p2 = (k / N) * (eps + (1 - delta) * ((N - k) / (N - 1))) if k > 0 else 0.0
        r = random.random()
        if r < p1:
            k += 1
        elif r < p1 + p2:
            k -= 1
        # sinon, rester à k
        states.append(k)
    return states

def estimate_stationary_distribution(states, N):
    """
    Estimer la distribution stationnaire à partir d'une séquence simulée d'états.
    """
    counts = np.bincount(states, minlength=N + 1)
    return counts / counts.sum()

def compute_stationary_from_matrix(P):
    """
    Calculer la distribution stationnaire en résolvant P^T π = π.
    """
    evals, evecs = np.linalg.eig(P.T)
    # localiser la valeur propre 1
    idx = np.argmin(np.abs(evals - 1.0))
    pi = np.real(evecs[:, idx])
    pi = pi / pi.sum()
    return pi

def main():

    N = int(input("Entrez le nombre de fourmis (N) : "))
    delta = float(input("Entrez le paramètre d'interaction δ : "))
    # calculer ε maximal basé sur ε < (1−δ)/(N−1)
    max_eps = (1 - delta) / (N - 1)
    print(f"Probabilité maximale de conversion spontanée ε < {max_eps:.6f}")
    eps = float(input(f"Entrez la probabilité de conversion spontanée ε (< {max_eps:.6f}) : "))
    steps_input = input("Entrez le nombre d'itérations de simulation [100000] : ").strip()
    steps = int(steps_input) if steps_input else 100000

    seed = 42 #pour reproductibilité

    random.seed(seed)
    np.random.seed(seed)

    # construire la matrice de transition
    P = transition_probabilities(N, eps, delta)
    print('\nMatrice de transition P :')
    print(P)

    # simuler et estimer la distribution stationnaire
    states = simulate_chain(N, eps, delta, num_steps=steps)
    # afficher la trajectoire complète des états visités
    print('\nTrajectoire de la chaîne :')
    print(states)
    # option d'échantillonnage pour le tracé
    choix = input("Tracer seulement chaque 50ème point ? (o/n): ").strip().lower()
    if choix == 'o':
        # sous-échantillonnage : ne garder qu'un point sur 50 et indexer de 0 à len(y_vals)-1
        y_vals = states[::50]
        x_vals = list(range(len(y_vals)))
    else:
        x_vals = list(range(len(states)))
        y_vals = states
    # tracer la trajectoire
    plt.figure()
    plt.plot(x_vals, y_vals, lw=1)
    plt.xlabel('Étape')
    plt.ylabel('Nombre de fourmis à la source noire (k)')
    # afficher les paramètres dans le titre du graphique
    params = f"N={N}, ε={eps}, δ={delta}, itérations={steps}"
    plt.title(f"Trajectoire de la chaîne de Markov\n{params}")
    plt.grid(True)
    plt.show()
    est_pi = estimate_stationary_distribution(states, N)
    print('\nDistribution stationnaire estimée (simulation) :')
    print(est_pi)

    # calculer la distribution stationnaire exacte
    pi = compute_stationary_from_matrix(P)
    print('\nDistribution stationnaire (vecteur propre) :')
    print(pi)

if __name__ == '__main__':
    main() 
