# bellman equations

## definitions

* RETURN (sum of the discounted rewards):
G_t = R_{t+1} + gamma*R_{t+2} + gamma^2*R_{t+3} + ...
* STATE VALUE -- expected return if we are in the state `s` at time `t`:
V(s) = E[G_t | S_t = s]

## decomposition of the equations, line by line
1. just definition from article
`V(s) = E[G_t | S_t = s]`

2. unwrap `G_t`
`G_t = R_{t+1} + gamma*R_{t+2} + gamma^2*R_{t+3} + ...`
`V(s) = E[ R_{t+1} + gamma*R_{t+2} + gamma^2*R_{t+3} + ... | S_t = s ]`

3. factor `gamma` out
`V(s) = E[ R_{t+1} + gamma*(R_{t+2} + gamma*R_{t+3} + ...) | S_t = s ]`

4. `G_{t+1}` is the same as thing that we have in bracket above
`G_{t+1}` == `(R_{t+2} + gamma*R_{t+3} + gamma^2*R_{t+4} + ...)`
so 
`V(s) = E[ R_{t+1} + gamma * G_{t+1} | S_t = s ]`

5. 
* split expectation (additions could split, pull `gamma` out)
 `E[ R_{t+1} | S_t = s ] + gamma * E[ G_{t+1} | S_t = s ]`
* now we focus on the second term `E[ G_{t+1} | S_t = s ]`
* it's average over all possible futures from the current `s`, average over (action, transition, future)
* we replace `G_{t+1}` with `V(S_{t+1})` not because we algebraically transform one to another, but because they give us same weighter sum when we average over the branches of possible future
`V(s) = E[ R_{t+1} + gamma * V(S_{t+1}) | S_t = s ]` 


# still open questions:
- "i understand most of it, by why do we sum all the probabilities again?", regard to `P_{ss'}^a = P(s' \vert s, a)  = \mathbb{P} [S_{t+1} = s' \vert S_t = s, A_t = a] = \sum_{r \in \mathcal{R}} P(s', r \vert s, a)`. actually the whole section "Model: Transition and Reward" and math in it should be revisited, with toy examples
- toy examples of usage *advantage function* 
- the whole section of bellman things, i think as well as markov, too mathy for the first read

---

+ on policy/off policy, eh?
  - algo learns by playing/experiencing the world on its own -- on-policy
  - algo learn from all sort of stuff -- datasets, human or external feedback -- off-policy
  - ez

>How the environment reacts to certain actions is defined by a model which we may or may not know.
what does it mean ".. may not know" for us? 
related to that:
>We may or may not know how the model works and this differentiate two circumstances:

to have the rules of how the world updates vs to have not?
*do i understand the mechanics of what happens when i act*

---

MDP -- Markov Decision Process

---

>The agent can stay in one of many states of the environment
state describes only AGENT state or state of the env too?

---

on each state, we could apply a value function, and with that, predict the expected amount of FUTURE reward

--- 

in RL, model = rules how the env updates 
* env -- world you interact with
* MODEL of the env -- how that world reacts on actions
  > the model is a descriptor of the environment


---

>This is known as one transition step, represented by a tuple (s, a, s’, r).
state -> action -> next state -> reward (based on the new state?)

---
P_{ss'}^a == "from state `s`, after action `a`, what's the chance of ending in `s'` -___-

--- 

policies:
determ -- map state to the action, e.g. hungry -- eat
stochastic -- action chosen by fix probs: rock-paper-scissors, explore maze
`\pi(a \vert s) = \mathbb{P}_\pi [A=a \vert S=s]`

---

future reward == *return*

---

how valuable given state or action called respectively state-value and action-value. diff between action-value and state-value is *advantage function* (A-value). we need this to compare diff actions on the same state 
