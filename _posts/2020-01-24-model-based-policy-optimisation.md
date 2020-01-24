# Model Based Policy Optimisation 

Here's the table of contents:

1. TOC
{:toc}


Total Variation (TV)

**Theorem 4.1.** Let the expected TV-distance between two transition distributions be bounded at each timestep by $\epsilon_{m}$ and the policy divergence be bounded by $\epsilon_{\pi}$. Then the true returns and model returns of the policy are bounded as:

$$$\nu\[\pi\] \geq \hat{\nu}\[\pi\] - \[\frac{2\gamma r{\text{max}}(\epsilon_{m} + 2\epsilon_{\pi})}}{(1-\gamma)^2} + \frac{4 r{\text{max}}\epsilon_{\pi}}{1-\gamma}\] $$$


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

