Skip to content
Chat history
You said:
Multiple state Markov chain demonstrated in python
ChatGPT said:

Here is a complete Python demonstration of a multiple-state (first-order) Markov chain, where:

    There are 4 states: A, B, C, D

    We define a transition probability matrix

    We simulate state transitions over time

    We visualize the state transitions

✅ 1. Define and Simulate the Markov Chain

import numpy as np
import matplotlib.pyplot as plt

# Define the states
states = ['A', 'B', 'C', 'D']
state_to_index = {state: i for i, state in enumerate(states)}

# Transition probability matrix (row: from, column: to)
P = np.array([
    [0.1, 0.6, 0.2, 0.1],  # From A
    [0.2, 0.1, 0.5, 0.2],  # From B
    [0.3, 0.2, 0.1, 0.4],  # From C
    [0.25, 0.25, 0.25, 0.25]  # From D
])

# Function to simulate the chain
def simulate_markov_chain(P, states, start_state, steps):
    current_state = start_state
    chain = [current_state]

    for _ in range(steps):
        i = state_to_index[current_state]
        next_state = np.random.choice(states, p=P[i])
        chain.append(next_state)
        current_state = next_state

    return chain

# Simulate the chain
np.random.seed(42)
simulated_chain = simulate_markov_chain(P, states, start_state='A', steps=50)

# Print simulation
print("Simulated Markov Chain:")
print(" -> ".join(simulated_chain))

✅ 2. Plot the State Frequencies

# Count occurrences
from collections import Counter
counts = Counter(simulated_chain)

# Bar plot
plt.figure(figsize=(6, 4))
plt.bar(counts.keys(), counts.values(), color='skyblue')
plt.title('State Visit Frequencies')
plt.xlabel('State')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

✅ 3. (Optional) Visualize the Transition Graph

import networkx as nx

def plot_transition_graph(P, states):
    G = nx.DiGraph()
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            weight = P[i][j]
            if weight > 0.05:
                G.add_edge(from_state, to_state, weight=round(weight, 2))

    pos = nx.circular_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=2000, font_size=12, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title('Markov Chain Transition Diagram')
    plt.show()

plot_transition_graph(P, states)

Summary

    You can adjust the number of states or transition matrix easily.

    This code includes simulation, frequency analysis, and a graphical visualization.

    Works without any external libraries except numpy, matplotlib, and networkx.

Let me know if you'd like:

    Second-order Markov chains

    Real-world application (e.g. weather, stock states)

    Transition matrix estimation from observed sequences

You said:
All three
ChatGPT said:

Great! Below is a complete Python demonstration of all three aspects you requested for a multi-state Markov chain:
✅ 1. First-Order Markov Chain (4 States) Simulation
🔧 Setup and Simulation Code

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter

# Define states
states = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
state_to_index = {state: i for i, state in enumerate(states)}

# Transition matrix (row: from state, col: to state)
P = np.array([
    [0.7, 0.2, 0.1, 0.0],  # Sunny
    [0.3, 0.4, 0.2, 0.1],  # Cloudy
    [0.2, 0.3, 0.4, 0.1],  # Rainy
    [0.1, 0.1, 0.2, 0.6],  # Snowy
])

def simulate_markov_chain(P, states, start_state, steps):
    current_state = start_state
    chain = [current_state]

    for _ in range(steps):
        i = state_to_index[current_state]
        next_state = np.random.choice(states, p=P[i])
        chain.append(next_state)
        current_state = next_state

    return chain

# Run simulation
np.random.seed(0)
sim_chain = simulate_markov_chain(P, states, start_state='Sunny', steps=50)
print(" → ".join(sim_chain))

✅ 2. Frequency Bar Chart

# Frequency count
counts = Counter(sim_chain)

# Plot
plt.figure(figsize=(6, 4))
plt.bar(counts.keys(), counts.values(), color='skyblue')
plt.title('State Visit Frequencies (First-Order)')
plt.xlabel('State')
plt.ylabel('Visits')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

✅ 3. Transition Diagram

def plot_transition_graph(P, states):
    G = nx.DiGraph()
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            prob = P[i][j]
            if prob > 0.05:
                G.add_edge(from_state, to_state, weight=round(prob, 2))

    pos = nx.circular_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=2000, font_size=12)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Transition Graph (First-Order)")
    plt.show()

plot_transition_graph(P, states)

✅ 4. Second-Order Markov Chain (Memory of 2)

from collections import defaultdict

# Simulate 2nd-order chain
def simulate_second_order(P2, state_pairs, start_pair, steps):
    current_pair = start_pair
    chain = [current_pair[0], current_pair[1]]

    for _ in range(steps):
        next_state_probs = P2.get(current_pair)
        if not next_state_probs:
            break
        next_states, probs = zip(*next_state_probs.items())
        next_state = np.random.choice(next_states, p=probs)
        chain.append(next_state)
        current_pair = (current_pair[1], next_state)

    return chain

# Build 2nd-order transition matrix manually
P2 = {
    ('Sunny', 'Sunny'): {'Sunny': 0.6, 'Cloudy': 0.3, 'Rainy': 0.1},
    ('Sunny', 'Cloudy'): {'Sunny': 0.3, 'Rainy': 0.5, 'Cloudy': 0.2},
    ('Cloudy', 'Rainy'): {'Rainy': 0.5, 'Snowy': 0.3, 'Cloudy': 0.2},
    ('Rainy', 'Snowy'): {'Snowy': 0.8, 'Rainy': 0.2},
    # Add more as needed
}

# Simulate and show chain
chain_2nd_order = simulate_second_order(P2, list(P2.keys()), ('Sunny', 'Sunny'), 30)
print("Second-order chain:")
print(" → ".join(chain_2nd_order))

✅ 5. Estimating Transition Matrix from Data

def estimate_first_order_matrix(data, states):
    n = len(states)
    counts = np.zeros((n, n))

    for (a, b) in zip(data[:-1], data[1:]):
        i = state_to_index[a]
        j = state_to_index[b]
        counts[i][j] += 1

    # Normalize to probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        P_est = np.nan_to_num(counts / counts.sum(axis=1, keepdims=True))

    return P_est

# Estimate from simulated chain
P_est = estimate_first_order_matrix(sim_chain, states)
print("Estimated transition matrix (First-Order):")
print(np.round(P_est, 2))

✅ 6. Real-World Example: Weather Model

You can interpret the states as real weather:

    Sunny, Cloudy, Rainy, Snowy

    Use weather observations to estimate P

    Forecast tomorrow's weather given today's using Markov dynamics

Would you like this as:

    ✅ A downloadable .zip with all code

    ✅ A live notebook (Colab / Jupyter)

    ✅ Or extended to support higher-order or continuous-time Markov models?

Let me know!
You said:
All in one code block
ChatGPT said:

Here is a complete Python demonstration of a multiple-state Markov chain simulation (first-order and second-order), including:

    State transition

    Frequency plot

    Transition diagram

    Estimation from data

All in one code block:

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter, defaultdict

# --- Define states and transition matrix ---
states = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
state_to_index = {s: i for i, s in enumerate(states)}

P = np.array([
    [0.7, 0.2, 0.1, 0.0],  # Sunny
    [0.3, 0.4, 0.2, 0.1],  # Cloudy
    [0.2, 0.3, 0.4, 0.1],  # Rainy
    [0.1, 0.1, 0.2, 0.6],  # Snowy
])

# --- First-order simulation ---
def simulate_first_order(P, states, start, steps):
    chain = [start]
    curr = start
    for _ in range(steps):
        i = state_to_index[curr]
        curr = np.random.choice(states, p=P[i])
        chain.append(curr)
    return chain

np.random.seed(0)
chain1 = simulate_first_order(P, states, 'Sunny', 50)
print("First-order chain:\n", " → ".join(chain1))

# --- Frequency plot ---
counts = Counter(chain1)
plt.figure(figsize=(6,4))
plt.bar(counts.keys(), counts.values(), color='skyblue')
plt.title('First-Order State Frequencies')
plt.xlabel('State'); plt.ylabel('Visits')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout(); plt.show()

# --- Transition graph ---
def plot_graph(P, states):
    G = nx.DiGraph()
    for i, s_from in enumerate(states):
        for j, s_to in enumerate(states):
            prob = P[i, j]
            if prob > 0.05:
                G.add_edge(s_from, s_to, weight=round(prob, 2))
    pos = nx.circular_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightgreen')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("First-Order Transition Diagram")
    plt.show()

plot_graph(P, states)

# --- Second-order Markov chain ---
P2 = {
    ('Sunny', 'Sunny'): {'Sunny': 0.6, 'Cloudy': 0.3, 'Rainy': 0.1},
    ('Sunny', 'Cloudy'): {'Sunny': 0.3, 'Rainy': 0.5, 'Cloudy': 0.2},
    ('Cloudy', 'Rainy'): {'Rainy': 0.5, 'Snowy': 0.3, 'Cloudy': 0.2},
    ('Rainy', 'Snowy'): {'Snowy': 0.8, 'Rainy': 0.2},
    ('Snowy', 'Snowy'): {'Sunny': 0.2, 'Snowy': 0.8},
}

def simulate_second_order(P2, start_pair, steps):
    chain = [start_pair[0], start_pair[1]]
    curr = start_pair
    for _ in range(steps):
        if curr not in P2:
            break
        choices, probs = zip(*P2[curr].items())
        next_state = np.random.choice(choices, p=probs)
        chain.append(next_state)
        curr = (curr[1], next_state)
    return chain

chain2 = simulate_second_order(P2, ('Sunny', 'Sunny'), 30)
print("\nSecond-order chain:\n", " → ".join(chain2))

# --- Estimate transition matrix from chain1 ---
def estimate_matrix(chain, states):
    n = len(states)
    M = np.zeros((n, n))
    for a, b in zip(chain[:-1], chain[1:]):
        i, j = state_to_index[a], state_to_index[b]
        M[i, j] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        P_est = np.nan_to_num(M / M.sum(axis=1, keepdims=True))
    return P_est

P_est = estimate_matrix(chain1, states)
print("\nEstimated transition matrix from first-order chain:")
print(np.round(P_est, 2))

🔧 Features Included:

    4-state Markov chain: Sunny, Cloudy, Rainy, Snowy

    First-order simulation & plot

    Transition diagram (using networkx)

    Second-order memory-based simulation

    Empirical transition matrix estimation from simulated data

Let me know if you want a GUI version, interactive dashboard, or Colab link!


ChatGPT can make mistakes. Check important info.
