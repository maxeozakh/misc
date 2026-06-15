from toy_rl import POLICY as pi, MODEL


print('policy', '\n', pi)


def expected_action_value(V, s, a, gamma):
  '''
  inner sum: Σ_{s',r} P(s',r|s,a) · (r + γ·V_t(s'))
  '''
  return sum(
    prob * (r + gamma * V[s_next])
    for s_next, r, prob in MODEL[(s, a)]
  )

def new_value(V, s, gamma):
  '''
  outer sum: Σ_a π(a|s) · [inner]
  '''
  return sum(
    pi_prob * expected_action_value(V, s, a, gamma)
    for a, pi_prob in pi[s].items()
  )

states = {s for (s, _) in MODEL}

V = {s: 0.0 for s in states}
g = 0.9

for _ in range(100):
  V = {s: new_value(V, s, g) for s in states}

print('policy eval', '\n', V)
