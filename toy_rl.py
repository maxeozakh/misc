import random 


MODEL = {
  ('sunny', 'stay'):
    [
      ('sunny', 1, 0.5), # hot 🥵
      ('sunny', -1, 0.1), # sunny and windy
      ('cloudy', 2, 0.3),
      ('rainy', 5, 0.1),
    ],
  ('sunny', 'go_out'):
    [

      ('sunny', -1, 0.4), # hot 🥵
      ('sunny', 1, 0.1),  # sunny and windy
      ('cloudy', 1, 0.4),
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

POLICY = {
  'sunny': {'stay': 0.8, 'go_out': 0.2},
  'rainy': {'stay': 0.99, 'go_out': 0.01},
  'cloudy': {'stay': 0.95, 'go_out': 0.05},
}
if __name__ == "__main__":
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
    a = random.choices(['stay', 'go_out'], weights=POLICY[s].values(), k=1)[0]
    return a

  def run_episode(s, n_steps, gamma, init_action = None):
    G = 0
    for i in range(n_steps):
      a = init_action if i == 0 and init_action else stochastic_policy(s)
      s, r = step(s, a)
      G += r * gamma ** i
    return G

  def run(s, n_episodes=10, n_steps = 10, gamma=1.0, init_action=None):
    '''
    gamma 1.0 by default, so w/o discount factor
    '''
    r = 0
    for _ in range(n_episodes):
      r += run_episode(s, n_steps, gamma, init_action)
    return r / n_episodes

  def advantage_fn(action_value, state_value):
    return action_value - state_value


  state = 'rainy'
  n_steps = 5
  n_episodes = 100
  state_value = run(state, n_steps=n_steps, n_episodes=n_episodes)


  # aka action-value
  init_action = 'go_out'
  Q_go_out = run(state, n_episodes=n_episodes, n_steps=n_steps, init_action=init_action)

  print(f'V({state}):', state_value)
  print(f'Q({state}, go_out):', Q_go_out)
  print(f'A({state}, go_out):', advantage_fn(Q_go_out, state_value))

  # # return aka accumulated discounted rewards
  # G = run_episode(state, n_steps=n_steps, gamma=0.9)
  # print('discounted reward', G)

  # policy evaluation is nested weighted average
  # over actions of policy -> over outcomes
