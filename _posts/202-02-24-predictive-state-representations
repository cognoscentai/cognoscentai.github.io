# Predictive State Representations


Predictive State Representations (PSR) offer an alternative internal state representation to RNNs in terms of the available observations. Predictive State Decoders (PSD) combine the state representations of PSRs and RNNs for sequential modelling. 

## Latent State Models

Sequential prediction problems are often defined in terms of Markov processes,

$P(s_{t+1}\vert s_{t},s_{t-1},...,s_{0}) = P(s_{t+1}\vert s_{t})$

In practise $s_{t}$ is only partially observed and we only have access to observations $o_{t}$. The machine learning problem is to find a model $f$ that uses the latest observation $o_{t}$ to recursively update an internal state $h_{t}$. Note that $h_{t}$ is the *learner's* internal state and $s_{t}$ is the state of the data-generating Markov process.  

## State Space Models and RNNs

The RNN model uses an internal state to make predictions $y_{t} = f(h_{t}, o_{t})$ and is trained by minimising a series of losses $l_{t}$ over each prediction,

$\underset{f}{\min} \mathcal{L} = \underset{f}{\min} \underset{t}{\sum} l_{t}(f(h_{t}, o_{t})$

The general difficult with this objective is that the recurrence with $f$ results in a non-convex, difficult optimisation (?).

## Predictive State Models

PSRs address the problem of finding an internal state by formulating the representation in terms of observable quantities. Instead of targeting a prediction loss as with RNNs, PSRs define a belief over the distribution of $k$ future observations, $g_{t} = [x_{t},...,x_{t+k}]$ given all past observations $u_{t}=[x_{0},...,x_{t-1}]$. The key assumption in PSRs is that the definition of state is equivalent to having *sufficient* information to predict everything about $g_{t}$ at time-step $t$, i.e. there is a bijective function the maps $p(s_{t}\vert u_{t})$, the distribution of latent state given the past, to $p(g_{t}\vert u_{t})$, the belief over future observations.

Significant improvement in learning PSRs was observed when sufficient feature functions $\phi$ (e.g. moments) map distributions $p(g_{t}\vert u_{t})$ to points in feature space $\mathbb{E}[\phi(g_{t})\vert u_{t}]$, e.g. $\mathbb{E}[\phi(g_{t})\vert u_{t}] = \mathbb{E}[g_{t}, g_{t}g_{t}^{T}\vert u_{t}]$ are sufficient statistics for a Gaussian distribution. A PSR model can be learned with the supervised objective,

$l_{p} = \underset{t}{\sum}\|\phi(g_{t+1}) - h_{t+1}\|,\text{ }h_{t+1}=f(h_{t}, o_{t})$

Call this the predictive state loss. 

The predictive state loss forms the basis of predictive state decoders (PSD). By minimising this loss, we force $h_{t}$ to match sufficient statistics of future observations $\mathbb{E}[\phi(g_{t})\vert u_{t}]$. We observe an empirical sample of the future $g_{t} = [x_{t},...,x_{t+k}]$ at each time-step by looking into the future in the training dataset or by waiting for streaming future observations. 

PSDs augment RNNs with the predictive state loss,

$\mathcal{L} = \mathcal{L}_{RNN} + \beta\mathcal{L}_{PSD}$

Where $\mathcal{L}_{PSD} = \underset{t}{\sum}\|F(h_{t})-\phi(o_{t+1}, o_{t+2},...)\|^{2}_{2}$.

## Hindsight Modelling

Hindsight modelling (HiMo) models the sequential problem with parametric form, 

$\psi_{\theta_{1}}(f(h_{t}), \phi_{\theta_{2}}(\tau_{t}^{+}))$

where $\tau_{t}^{+} = s_{t+1},s_{t+2},...,s_{t+k}$. This formulation forces information about the future trajectory through a vector valued *hindsight* function $\phi$. The HiMo architecture can summarised as,

$v^{+}(h_{t}, h_{t+k}; \theta) = \psi_{\theta_{1}}(\overline{h_{t}}, \phi_{\theta_{2}}(\overline{h_{t+k}})$,

$v^{m}(h_{t}; \nu) = \psi_{\nu_{1}}(\overline{h_{t}}, \overline{\hat{\phi}_{\nu_{2}}(h_{t})})$.

where bar notation indicates that a quantity is treated as non-differentiable (i.e the gradient is stopped). The HiMo loss is defined as,

$\mathcal{L}(\theta, \nu) = \mathcal{L}_{v}(\nu) + \alpha\mathcal{L}_{v^{+}}(\theta)+\beta\mathcal{L}_{\text{model}}(\nu)$,

where $\mathcal{L}_{\text{model}}(\nu) = \mathbb{E}_{h,\tau^{+}}[\|\hat{\phi}_{\nu_{2}}(\tau^{+}) - \phi(h)\|^{2}_{2}]$.

This is a generalisation of PSD as $\phi$ is a learned representation and not assumed.

