


# K Learning

In order for an RL algorithm to be statistically efficient, it must consider the value of information. To do this an RL algorithm should use posterior estimates for unknown problem parameters and use this distribution to drive an efficient trade-off between exploitation and exploration.

## Thompson Sampling

Thompson sampling (TS) randomly selects a policy according to the probability that it is the optimal policy, conditioned upon the data seen prior to that episode. Define the binary random variable $\mathcal{O}_{t}(s,a)$, where $\mathcal{O}_{t}(s,a) = 1$ denotes the event that action $a$ is optimal for state $s$ at time $t$. The TS policy for episode $l$ is thus given by,

$$\pi_{TS}\sim \mathbb{P}(\mathcal{O}\vert \mathcal{F}_{l})$$

where $\mathbb{P}(\mathcal{O}\vert \mathcal{F}_{l})$ is the probability over binary optimality variables given $\mathcal{F}_{l}$, the data gathered prior to episode $l$.

Table 1 describes one approach to performing such sampling implicitly, by maintaining an explicit model over MDP parameters. This algorithm has been scaled by taking *approximate* posterior samples via randomised value functions. It is not yet clear under which settings these approximations should be expected to perform well.

---

**Table 1** Model-Based Thompson Sampling

---

Before episode $l$						Sample $M_{l} = (\mathcal{S}, \mathcal{A}, \mathcal{R}^{l}, \mathcal{P}^{l}, \mathcal{H}, \rho)\sim \phi\vert \mathcal{F}_{l}$

Bellman equation					  $Q^{l}_{t} = \mu^{l}(s,a) + \sum_{s^{\prime}}\mathcal{P}^{l}V^{l}_{t+1}(s^{\prime})$

Policy 										   $\pi_{t}^{TS}(s,a)\in \text{argmax }Q^{l}_{t}(s,a)$

---

## RL as Inference

Another scalable approach to posterior inference over probability of optimality is known as *RL as inference*. These algorithms model the probability of optimality as,

$\mathbb{P}(\mathcal{O}_{t}(s,a)\vert\pi_{t}(s,a))\propto \exp\left(\underset{(s,a)\in\tau_{t}(s,a)}{\sum}\beta\mathbb{E}^{l}\mu(s^{\prime},a^{\prime})\right)$

for some $\beta\gt 0$, where $\tau_{t}(s,a)$ is a trajectory starting from $(s,a)$ at timestep $t$, and $\mathbb{E}^{l}$ denotes the expectation under the posterior at episode $l$. Applying inference to this probability lead to RL algorithms with *soft* Bellman updates and added entropy regularisation. Table 2 describes the general structure. The problem with *RL as inference* is the resultant posterior does not bear a close relationship to agent's epistemic probability that $(s,a,t)$ is optimal.

---

**Table 2** Soft Q-Learning

---

Bellman equation					 $Q_{t}(s,a)=\mathbb{E}^{l}\mu(s,a)+\sum_{s^{\prime}}\mathbb{E}^{l}\mathcal{P}(s^{\prime},s,a)V_{t+1}(s^{\prime})$

​													 $V_{t}(s)=\beta^{-1}\log\sum_{a}\exp\beta Q_{t}(s,a)$

Policy 										  $\pi^{SQ}_{t}(s,a)\propto \exp\beta Q_{t}(s,a)$

---

## K-Learning

K_learning [^1] is an approximate inference scheme with connections to TS and *RL as inference*, that develops a coherent notion of optimality. Consider the approximate conditional optimality probability,

$\mathbb{P}(\mathcal{O}_{t}(s,a)\vert Q^{M,\star}_{t})\propto\exp\beta Q^{M,\star}_{t}(s,a)$

for some $\beta\gt 0$. We can marginalise over possible Q-values yielding,

$\mathbb{P}(\mathcal{O}_{t}(s,a))=\int\mathbb{P}(\mathcal{O}_{t}(s,a)\vert Q^{M,\star}_{t})d\mathbb{P}(Q^{M,\star}_{t}(s,a))\propto\exp G^{Q}_{t}(s,a,\beta)$

where $G^{Q}_{t}(s,a,\beta)$ denotes the cumulant generating function of the random variable $Q^{M,\star}_{t}(s,a)$. Clearly K-learning and *RL as inference* are similar. The difference is the inclusion of an integral with respect to the posterior over $Q^{M,\star}_{t}(s,a)$, which includes the epistemic uncertainty explicitly.

Given the approximation to the posterior probability of optimality we could sample from it as our policy, as done in TS. However, the computation of the cumulant generating function, $G^{Q}_{t}(s,a,\beta)$, is non-trivial. It can be shown that an upper bound to the cumulant generating function can be computed by solving a particular *soft* Bellman equation [^1]. The resulting K-values, denoted $K_{t}(s,a)$, are optimistic for the expected optimal Q-values. For a sequence $\{\beta_{l}\}$ we have,

$K_{t}(s,a)\geq \beta^{-1}_{l}G^{Q}_{t}(s,a,\beta)\geq \mathbb{E}^{l}Q^{M,\star}_{t}(s,a)$.

Following a Boltzmann policy over these K-values satisfies a Bayesian regret bound which matches the current best bound for TS up to logarithmic factors under the same assumptions. Table 3 summarises the K-learning algorithm.

---

**Table 3** K-Learning

---

Before episode $l$ 					Calculate $\beta_{l}=\beta\sqrt{l}$

Bellman equation 				 $K_{t}(s,a)=\beta^{-1}_{l}G^{\mu}(s,a,\beta_{l})+\sum_{s^{\prime}}\mathbb{E}^{l}\mathcal{P}(s^{\prime},s,a)V^{K}_{t+1}(s^{\prime})$

​												  $V^{K}_{t}(s)=\beta^{-1}_{l}\log\sum_{a}\exp\beta_{l}K_{t}(s,a)$

Policy 									  $\pi^{K}_{t}(s,a)\propto\exp\beta_{l}K_{t}(s,a)$

---







[^1]: Brendan O’Donoghue. Making Sense of Reinforcement Learning and Probabilistic Inference. [*arXiv preprint arXiv:2001.00805*](https://arxiv.org/pdf/2001.00805.pdf), 2020.
[^2]: Brendan O’Donoghue. Variational Bayesian reinforcement learning with regret bounds. [*arXiv preprint arXiv:1807.09647*](https://arxiv.org/pdf/1807.09647.pdf), 2018. 
