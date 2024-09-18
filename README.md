# POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.

## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

## States
The environment has 7 states:
    Two Terminal States: G: The goal state & H: A hole state.
    Five Transition states / Non-terminal States including S: The starting state.

## Actions
The agent can take two actions:
    R: Move right.
    L: Move left.

## Transition Probabilities
The transition probabilities for each action are as follows:
    50% chance that the agent moves in the intended direction.
    33.33% chance that the agent stays in its current state.
    16.66% chance that the agent moves in the opposite direction.

For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

## Rewards
The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

## POLICY ITERATION ALGORITHM
Include the steps involved in policy iteration algorithm

## POLICY IMPROVEMENT FUNCTION
### Name : POOJA A
### Register Number: 212222240072
```
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Write your code here to implement policy improvement algorithm
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob, next_state,reward, done in P[s][a]:
          Q[s][a]+= prob*(reward+gamma*V[next_state]*(not done))
          new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return new_pi
```

## POLICY ITERATION FUNCTION
```
def policy_iteration(P, gamma=1.0,theta=1e-10):
  random_actions=np.random.choice(tuple(P[0].keys()),len(P))
  pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
  while True:
    old_pi = {s:pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P,gamma,theta)
    pi = policy_improvement(V,P,gamma)
    if old_pi == {s:pi(s) for s in range(len(P))}:
      break
  return V,pi
```

## OUTPUT:
Mention the optimal policy, optimal value function , success rate for the optimal policy.

## RESULT:
Thus, Python program is developed to find the optimal policy for the given MDP using the policy iteration algorithm.
