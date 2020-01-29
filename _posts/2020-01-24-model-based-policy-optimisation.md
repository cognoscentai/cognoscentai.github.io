# Model Based Policy Optimisation 

Here's the table of contents:

1. TOC
{:toc}

Model-based policy optimisation (MBPO) defines a general algorithm for guaranteeing monotonic policy improvement in policy search reinforcement learning. This algorithm resembles or subsumes several prior algorithms and provides a framework for theoretical analysis. From the theoretical analysis the authors design a practical model-based reinforcement learning that achieves state-of-the-art performance on continuous control benchmarks, comparable to model-free methods, with the sample efficiency of a model-based method. 

In Algorithm 1, we give the general algorithm for guaranteed monotonic policy improvement.



---

**Algorithm 1** Monotonic Model-Based Policy Optimisation

---

1. Initialise policy $\pi(a\vert s)$, predictive model $p_{\theta}(s^{\prime}, r\vert s,a)$, empty dataset $\mathcal{D}$.
2. **for** $N$ epochs **do**
3.     Collect data with $\pi$ in environment: $\mathcal{D}=\mathcal{D}\cup\{(s_{i},a_{i},s^{\prime}_{i},r_{i})\}$
4.     Train model $p_{\theta}$ on $\mathcal{D}$ via maximum likelihood: $\theta\leftarrow \underset{\theta}{\text{argmax}}E_{\mathcal{D}}[\text{log}p_{\theta}(s^{\prime}, r\vert s,a)]$ 
5.     Optimise policy under predictive model: $\pi\leftarrow \underset{\pi}{\text{argmax}}\hat{\nu}\left[\pi\right] - C(\epsilon_{m}, \epsilon_{\pi})$

---

To guarantee monotonic improvement for a model-based method, we construct a bound of the form:

$\nu[\pi] \geq \hat{\nu}[\pi] - C$.

$\nu[\pi]$ denotes the returns of the policy in the true MDP, $\hat{\nu}[\pi]$ denotes the returns of the policy under the model. The bound guarantees that as long we improve returns by at least $C$ under the model, we are guaranteed improvement on the true MDP.

The gap between true return and model returns, $C$, can be expressed in terms of two model errors:

1. generalisation error due to sampling, $\epsilon_{m}$.
2. distribution shift error due to the updated policy encountering states not observed during model training, $\epsilon_{\pi}$.

$\epsilon_{m}$ can be quantified using PAC generalisation bounds for supervised learning. This bounds the difference between expected loss and empirical loss by a constant with high probability, $\epsilon_{m} = \underset{t}{\text{max}} E_{s \sim \pi_{D,t}} \left[D_{TV}\left(p\left(s^{\prime}\vert s,a\right)\Vert \hat{p}\left(s^{\prime}\vert s,a\right)\right)\right]$. This can be estimated by measuring the validation loss of the model on the time-dependent state distribution of the data-collecting policy, $\pi_{D}$.

$\epsilon_{\pi}$ can be quantified by, $\epsilon_{\pi} \geq \underset{s}{\text{max}}D_{TV}\left(\pi\Vert\pi_{D}\right)$, the maximum total-variation distance between of the policy between iterations. In practise, we measure the KL divergence between policies, which can be related to $\epsilon_{\pi}$ by Pinskers inequality, $D_{TV}\left(P\Vert Q\right) \leq \sqrt{\frac{1}{2}D_{KL}\left(P\Vert Q\right)}$.

The following theorem defines $C$ in terms of $\epsilon_{m}$ and $\epsilon_{\pi}$.

**Theorem 4.1.** *Let the expected TV-distance between two transition distributions be bounded at each timestep by $\epsilon_{m}$ and the policy divergence be bounded by $\epsilon_{\pi}$. Then the true returns and model returns of the policy are bounded as*:

$\nu\left[\pi\right] \geq \hat{\nu}\left[\pi\right] - \left[\frac{2\gamma r_{\text{max}}(\epsilon_{m} + 2\epsilon_{\pi})}{(1-\gamma)^2} + \frac{4 r_{\text{max}}\epsilon_{\pi}}{1-\gamma}\right].$

Theorem 4.1 implies that as long as we can improve the returns under the model $\hat{\nu}[\pi ]$ by more than $C(\epsilon_{m}, \epsilon_{\pi})$, we guarantee improvement under the true returns.

**Theorem 4.2.** *Given returns $\nu^{\text{branch}}[\pi]$ from the k-branched rollout method,*

$\nu\left[\pi\right] \geq \hat{\nu}^{\text{branch}}\left[\pi\right] - 2r_{\text{max}}\left[\frac{\gamma^{k+1} \epsilon_{\pi}}{(1-\gamma)^2} + \frac{\gamma^{k}+2}{1-\gamma}\epsilon_{\pi} + \frac{k}{1-\gamma}\left(\epsilon_{m} + 2\epsilon_{\pi}\right)\right].$

Theorem 4.2 is a pessimistic bound as it assumed access to only model error $\epsilon_{m}$ on the distribution of the most recent data-collecting policy $\pi_{D}$ and approximated error on the model error for the distribution of data for the current policy $\pi$, $\epsilon_{m} + 2\epsilon_{\pi}$. A tighter bound is developed by approximating the model error on the distribution of the current policy $\pi$, which we denote as $\epsilon_{m^{\prime}}$ and define as, $\hat{\epsilon}_{m^{\prime}}(\epsilon_{\pi}) \approx \epsilon_{m} + \epsilon_{\pi}\frac{d\epsilon_{m^{\prime}}}{d\epsilon_{\pi}}$

**Theorem 4.3.** *Under the k-branched rollout method, using model error under the updated policy $\epsilon_{m^{\prime}}\geq \underset{t}{\text{max}} E_{s\sim\pi_{D},t}\left[D_{TV}(p(s^{\prime}\vert s,a)\Vert \hat{p}(s^{\prime}\vert s,a)))\right]$, we have,*

$\nu\left[\pi\right] \geq \hat{\nu}^\text{branch}\left[\pi\right] - 2r_{\text{max}}\left[\frac{\gamma^{k+1} \epsilon_{\pi}}{(1-\gamma)^2} + \frac{\gamma^{k}+2}{1-\gamma}\epsilon_{\pi} + \frac{k}{1-\gamma}\left(\epsilon_{m^{\prime}}\right)\right].$

While this bound appears similar to Theorem 4.2, the important difference is that this version motivates model usage, since $k^{*} = \underset{k}{\text{argmin}} \left[\frac{\gamma^{k+1} \epsilon_{\pi}}{(1-\gamma)^2} + \frac{\gamma^{k}+2}{1-\gamma}\epsilon_{\pi} + \frac{k}{1-\gamma}\left(\epsilon_{m^{\prime}}\right)\right] > 0$ for sufficiently low $\epsilon_{m^{\prime}}$.

This motivates the following model-based reinforcement learning algorithm:

---

**Algorithm**

---

1. Initialise policy $\pi_{\phi}$, predictive model $p_{\theta}$, environment dataset $\mathcal{D}_{\text{env}}$, model dataset $\mathcal{D}_{model}$
2. **for** $N$ epochs **do**
3.  Train model $p_{\theta}$ on $\mathcal{D}_{\text{env}}$ via maximum likelihood
4.  **for** $E$ steps **do **
5.   Take action in environment according to $\pi_{\phi}$; add to $\mathcal{D}_{\text{env}}$
6.   **for** $M$ model rollouts **do**
7.    Sample $s_{t}$ uniformly from $\mathcal{D}_{\text{env}}$
8.    Perform $k$-step model rollout starting from $s_{t}$ using policy $\pi_{\phi}$; add to $\mathcal{D}_{\text{model}}$
9.   **for** $G$ gradient updates **do**
10.    Update policy parameters on model data: $\phi \leftarrow \phi - \lambda_{\pi}\hat{\nabla}_{\phi}J_{\phi}\left(\phi, \mathcal{D}_{\text{model}}\right)$

---

The key features of Algorithm 1 are:

1.  $k$-length model rollouts from replay buffer states, $k$ controls $\epsilon_{m}$. 
2. a fixed number of policy update steps, $G$, which controls $\epsilon_{\pi}$.

Another advantage is that even with short rollout length, $k$, many such rollouts can be performed to yield a large set of model samples for policy optimisation. This large set enables many more gradient steps per environment sample than is typically stable in model-free algorithms.






## Footnotes

[^1]: This is the footnote.

