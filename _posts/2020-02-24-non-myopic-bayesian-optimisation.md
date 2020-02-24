# Non-Myopic Bayesian Optimisation



In this post we consider non-myopic bayesian optimisation [^1]. Consider a sequential decision making (SDM) problem with a finite horizon, $T$, with action space, $\mathcal{A}$, state space $\mathcal{S}$, transition model $P(s^{\prime}\vert s,a)$ and reward function $R(s^{\prime}\vert s,a)$. Let $Q_{k}(a\vert s)$ be the expected reward of taking action $a$ at state $s$ when there are $k$ steps remaining assuming all later actions are optimal. We assume no discounting for future reward, which is standard in bayesian optimisation (BO) and bayesian quadrature (BQ). The Bellman equation is defined as follows:

$Q_{k}(a\vert s)=\mathbb{E}_{s^{\prime}}\left[R(s^{\prime}\vert s,a)\right] +\mathbb{E}_{s^{\prime}}\left[\underset{a^{\prime}}{\max}Q_{k-1}(a^{\prime}\vert s^{\prime})\right]$ 

where $s^{\prime}\sim P(s^{\prime}\vert s,a)$. The optimal (expected) policy is,

$a^{\star}=\underset{a}{\text{argmax }}Q_{T-i}(a\vert s_{i})$

The optimal policy is intractable for any large horizon. A tractable approximation is to simply limit the horizon to small $l$, e.g. $l=1 \text{ or } 2$ . This is called *$l$-step lookahead*, and is computationally efficient but *myopic* in the sense that we severely limit our view of the future. We don't consider future rewards and thus make suboptimal tradeoffs between exploration and exploitation. 

## Non-Myopic Approximation via a One-Step Optimal Batch Policy

Suppose $T$ actions $A=\{a_{1},...,a_{T}\}$ must be simultaneously decided from the initial state $s$, the expected reward would be

$Q(A\vert s)=\mathbb{E}_{S}\left[R(S\vert s,A)\right]$ 

where $S=\{s_{1},...,s_{T}\}\sim P(S\vert s,A)$. Decomposing $A$ into $a_{i}$ and $A_{-i}=A\backslash \{a_{i}\}$ we obtain

$Q(A\vert s)=\mathbb{E}_{s_{i}}\left[R(s_{i}\vert s,a)\right]+\mathbb{E}_{s_{i}}\left[Q(A_{-i}\vert s_{i})\right]$

where $s_{i}$ is the state after taking action $a_{i}$. Let $A^{\star}\in\text{argmax}_{A}\text{ }Q(A\vert s)$ be an optimal batch of actions. For any point $a^{\star}_{i}\in A^{\star}$, it follows that

$\mathbb{E}_{s^{\star}_{i}}\left[Q(A^{\star}_{-i}\vert s^{\star}_{i})\right] \equiv \underset{A_{-i}}{\max}\mathbb{E}_{s^{\star}_{i}}\left[Q(A_{-i}\vert s^{\star}_{i})\right]$

Therefore choosing a point $a^{\star}\in A^{\star}$ is equivalent to solving the following optimisation problem

$a^{\star}\in\text{argmax}\left(\mathbb{E}_{s^{\prime}}\left[R(s^{\prime}\vert s,a)\right]+\underset{A^{\prime}:\vert A^{\prime}\vert=T-1}{\max}\mathbb{E}_{s^{\prime}}\left[Q(A^{\prime}\vert s^{\prime})\right]\right)$

Note the resemblance of the this expression with the Bellman equation. There are two differences

1. the expectation and maximisation are exchanged in the future reward term,
2. the adaptive expected reward is replaced by a non-adaptive counterpart, i.e. $Q_{k}(a\vert s)$ by $Q(A\vert s)$ with $\vert A\vert = k$.

Due to these differences this objective is a *lower bound* of the *true expected utility*
$$
\underset{A^{\prime}:\vert A^{\prime}\vert=T-1}{\max}\mathbb{E}_{s^{\prime}}\left[Q(A^{\prime}\vert s^{\prime})\right]\leq \mathbb{E}_{s^{\prime}}\left[\underset{A^{\prime}:\vert A^{\prime}\vert=T-1}{\max}Q(A^{\prime}\vert s^{\prime})\right]
\leq \mathbb{E}_{s^{\prime}}\left[\underset{a^{\prime}}{\max}Q(a^{\prime}\vert s^{\prime})\right]
$$
An open question is the tightness of this bound, which is closely related to the *adaptivity gap* [^2].

Note that this optimisation problem maximises the Bellman equation if the transition model becomes stationary, i.e. $P(s^{\prime}\vert s_{j},a)=P(s^{\prime}\vert s_{j},a)$ for $a\in\mathcal{A}$ and $j\geq i$.

Algorithm 1 summarises the general framework for non-myopic approximation of the optimal solution to finite-horizon SDM problem.

---

**Algorithm 1** 

---

**input** action space $\mathcal{A}$, state space $\mathcal{S}$, transition model $P(s^{\prime}\vert s,a)$, reward function $R(s^{\prime}\vert s,a)$, horizon $T$

**output** $\mathcal{D}$, a sequence of actions and observations

**for** $k\leftarrow 0$ **to** $T-1$ **do**

​		Compute the optimal batch $A^{\star}$ if size $T-k$

​        Pick an action $a^{\star}\in A^{\star}$ and observe state $s^{\star}$

​        Augment $\mathcal{D}=\mathcal{D}\cup(a^{\star},s^{\star})$

---

## Non-Myopic Bayesian Optimisation

Consider a maximisation problem 

$x^{\star}=\text{argmax}_{x\in\mathcal{X}}\text{ }f(x)$.

Suppose we have a budget of $T$ function evaluations. Once the budget has been expended, the point with the highest observed value is recommended as the maximiser of $f$. Formally, our goal is to sequentially select a set $X=\{x_{1},...,x_{T}\}$ of points from $\mathcal{X}$ such that $\max_{i} \{y_{i}\}$ is maximised, where $y_{i}=f(x_{i})$.

In the SDM formulation, the state space is any set of observations $\mathcal{D}=\{(x_{i},y_{i})\}$, the action space is $\mathcal{X}$de. For example, at iteration $k$, the trajectory of observations is $\mathcal{D}_{k}=\mathcal{D}_{0}\cup\{(x_{i},y_{i})\}^{k}_{i=1}$, where $\mathcal{D}_{0}$ is a set of initial observations. Suppose the best initial observed value is $y_{0}$, define the utility function as the improvement over $y_{0}$

$u(X)=\left(\underset{x_{i}\in X}{\max }f(x_{i})-y_{0}\right)^{+}$

where $a^{+}=\max(a,0)$. 

Defining the utility as improvement allows us to write the expected utility in the form of a Bellman equation

$EI_{k}(x)=EI_{1}(x)+\mathbb{E}_{y}\left[\max_{x^{\prime}}EI_{k-1}(x^{\prime}\vert x,y)\right]$

 where $EI_{k}(x)$ is the expected improvement of $k$ of decisions starting from $x$, and $EI_{k-1}(x^{\prime}\vert x,y)$ is an expectation taken over the posterior belief of $f$ after further conditioning on the observation $(x,y)$ and replacing $y_{0}$ with $\max(y_{0},y)$.

This can be derived as follows. The Bayesian optimal policy selects a point maximising the expected utility

$x^{\star}=\underset{x}{\text{argmax}}\mathbb{E}\left[u(X)\vert x\right]$

where the expectation is taken w.r.t. the posterior belief of $f$ conditioned on all observations so far. 

When $T=1$, i.e. there is only one evaluation left, the optimal policy degenerates to the simplest case known as *expected improvement* (EI)

$x^{\star}=\underset{x}{\text{argmax }}EI_{1}(x)\equiv\mathbb{E}\left[(f(x)-y_{0})^{+}\right]$.

We use $EI_{k}(x)$ to denote expected improvement of $k$ sequential evaluations starting from $x$ and $EI_{k}(x^{\prime}\vert x,y)$ to indicate that the expectation is taken over the posterior belief after further conditioning on observation $(x,y)$ with $y_{0}$ replaced by $\max(y_{0},y)$.

Now consider $T=2$. Starting from location $x$, the improvement of the next two evaluations depends on three random variables, $y\equiv f(x)$, the next evaluation location $x^{\prime}$ and its value $y^{\prime}\equiv f(x^{\prime})$.



[^1] Shali Jiang, Henry Chai, Javier Gonzalez, Roman Garnett. Efficient nonmyopic Bayesian optimization and quadrature. [*arXiv preprint arXiv:1909.04568*](https://arxiv.org/pdf/1909.04568.pdf), 2019.

[^2] Shali Jiang, Gustavo Malkomes, Benjamen Moseley, Roman Garnett. Efficient nonmyopic batch active search. [*arXiv preprint arXiv:1811.08871*](https://arxiv.org/pdf/1811.08871.pdf), 2018.
