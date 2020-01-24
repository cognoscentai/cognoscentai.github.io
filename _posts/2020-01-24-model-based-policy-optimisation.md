# Model Based Policy Optimisation 

Here's the table of contents:

1. TOC
{:toc}


Total Variation (TV)

**Theorem 4.1.** *Let the expected TV-distance between two transition distributions be bounded at each timestep by $\epsilon_{m}$ and the policy divergence be bounded by $\epsilon_{\pi}$. Then the true returns and model returns of the policy are bounded as*:

$\nu\left[\pi\right] \geq \hat{\nu}\left[\pi\right] - \left[\frac{2\gamma r_{\text{max}}(\epsilon_{m} + 2\epsilon_{\pi})}}{(1-\gamma)^2} + \frac{4 r_{\text{max}}\epsilon_{\pi}}{1-\gamma}\right].$

Theorem 4.1 implies that as long as we can improve the returns under the model $\hat{\nu}\[\pi\]$ by more than $C(\epsilon_{m}, \epsilon_{\pi})$, we guarantee improvement under the true returns.

**Theorem 4.2.** *Given returns $\nu^{\text{branch}}\[\pi\]$ from the k-branched rollout method,*

$\nu\left[\pi\right] \geq \hat{\text{branch}\left[\pi\right] - 2r_{\text{max}}\left[\frac{\gamma^{k+1} \epsilon_{\pi}}{(1-\gamma)^2} + \frac{\gamma^{k}+2}{1-\gamma}\epsilon_{\pi} + \frac{k}{1-\gamma}\left(\epsilon_{m} + 2\epsilon_{\pi}\right)\right].$

**Theorem 4.3.** *Under the k-branched rollout method, using model error under the updated policy $\epsilon_{m^{\prime}}\geq \underset{t}{\text{max}} E_{s\sim\pi_{D},t}\left[D_{TV}(p(s^{\prime}\vert s,a)\Vert \hat{p}(s^{\prime}\vert s,a)))\right]$, we have,*

$\nu\left[\pi\right] \geq \hat{\text{branch}\left[\pi\right] - 2r_{\text{max}}\left[\frac{\gamma^{k+1} \epsilon_{\pi}}{(1-\gamma)^2} + \frac{\gamma^{k}+2}{1-\gamma}\epsilon_{\pi} + \frac{k}{1-\gamma}\left(\epsilon_{m^{\prime}}\right)\right].$

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

