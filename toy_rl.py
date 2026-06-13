import random 


MODEL = {
  ('sunny', 'stay'):
    [
      ('sunny', 1, 0.4), # hot 🥵
      ('sunny', -1, 0.1), # sunny and windy
      ('cloudy', 2, 0.4),
      ('rainy', 3, 0.1),
    ],
  ('sunny', 'go_out'):
    [

      ('sunny', -1, 0.4), # hot 🥵
      ('sunny', 1, 0.1),  # sunny and windy
      ('cloudy', 2, 0.4),
      ('rainy', -2, 0.1),
    ],
  ('cloudy', 'stay'):
    [
      ('sunny', 3, 0.4), # hot 🥵
      ('sunny', -1, 0.1), # sunny and windy
      ('cloudy', 1, 0.4),
      ('rainy', 2, 0.1),
    ],
  ('cloudy', 'go_out'):
    [
      ('sunny', -5, 0.4), # hot 🥵
      ('sunny', 1, 0.1),  # sunny and windy
      ('cloudy', 2, 0.4),
      ('rainy', -3, 0.1),
    ],
  ('rainy', 'stay'):
    [
      ('sunny', 1, 0.1), # hot 🥵
      ('sunny', -1, 0.1), # sunny and windy
      ('cloudy', 1, 0.3),
      ('rainy', 10, 0.5),
    ],
  ('rainy', 'go_out'):
    [
      ('sunny', 1, 0.1), # hot 🥵
      ('sunny', 1, 0.1),  # sunny and windy
      ('cloudy', 3, 0.3),
      ('rainy', -10, 0.5),
    ],
}


# planning needs to be aware of state evolution
def P_next(model, s, a, s_next):
  '''
  P_{ss'}^a == "from state `s`, after action `a`,
  what's the chance of ending in `s'`
  '''
  return sum(
    probability for state_p, reward, probability in model[s, a]
    if state_p == s_next
  )
p = P_next(MODEL, 'sunny', 'go_out', 'sunny')

# planning needs to be aware of expected reward
def R(model, s, a):
  '''
  predicts the next reward triggered by one action
  aka expected IMMEDIATE reward
  without discounting/future rewards yet
  '''
  return sum(
    reward * probability for _, reward, probability in model[(s, a)]
  )

r_stay = R(MODEL, 'sunny', 'stay')
r_go_out = R(MODEL, 'sunny', 'go_out')
assert r_stay > r_go_out # 🏡🏡🏡🏡🏡🏡🏡🏡

def step(s, a):
  outcomes = MODEL[(s, a)]
  weights = [probs for _, _, probs in outcomes]
  idx = random.choices(range(len(outcomes)), weights,k=1)[0]
  next_state, r, _ = outcomes[idx]

  return next_state, r

def stochastic_policy(s):
  '''
  non-determenistic action based on state
  '''
  p = {
    'sunny': {'stay': 0.8, 'go_out': 0.2},
    'rainy': {'stay': 0.99, 'go_out': 0.01},
    'cloudy': {'stay': 0.95, 'go_out': 0.05},
  }

  a = random.choices(['stay', 'go_out'], weights=p[s].values(), k=1)[0]
  return a

n_steps = 10
reward = 0
s = 'sunny'

for i in range(n_steps):
  a = stochastic_policy(s)
  s, r = step(s, a)
  reward += r

# without discount
print('avg reward per step', reward / n_steps)

# return aka accumulated discounted rewards
G = 0
gamma = 0.90
s = 'sunny'
for i in range(n_steps):
  a = stochastic_policy(s)
  s, r = step(s, a)
  G += r * gamma ** i
print('discounted reward', G)
