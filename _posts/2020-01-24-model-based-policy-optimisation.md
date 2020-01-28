# Model Based Policy Optimisation 

Here's the table of contents:

1. TOC
{:toc}

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




## Basic setup

Jekyll requires blog post files to be named according to the following format:

`YEAR-MONTH-DAY-filename.md`

Where `YEAR` is a four-digit number, `MONTH` and `DAY` are both two-digit numbers, and `filename` is whatever file name you choose, to remind yourself what this post is about. `.md` is the file extension for markdown files.

The first line of the file should start with a single hash character, then a space, then your title. This is how you create a "*level 1 heading*" in markdown. Then you can create level 2, 3, etc headings as you wish but repeating the hash character, such as you see in the line `## File names` above.

## Basic formatting

You can use *italics*, **bold**, `code font text`, and create [links](https://www.markdownguide.org/cheat-sheet/). Here's a footnote [^1]. Here's a horizontal rule:

---

## Lists

Here's a list:

- item 1
- item 2

And a numbered list:

1. item 1
1. item 2

## Boxes and stuff

> This is a quotation

{% include alert.html text="You can include alert boxes" %}

...and...

{% include info.html text="You can include info boxes" %}

## Images

![](/images/logo.png "fast.ai's logo")

## Code

General preformatted text:

    # Do a thing
    do_thing()

Python code and output:

```python
# Prints '2'
print(1+1)
```

    2

## Tables

| Column 1 | Column 2 |
|-|-|
| A thing | Another thing |

## Footnotes

[^1]: This is the footnote.

