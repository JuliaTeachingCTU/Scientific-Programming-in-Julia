# How to use admonitions

Documenter.jl provides five special styles for admonitions and one style for custom admonition types

| Admonition type | html class                      |
| :--             | :--                             |
| `info`          |`"admonition is-info"`           |
| `compat`        |`"admonition is-compat"`         |
| `dange`         |`"admonition is-danger"`         |
| `warning`       |`"admonition is-warning"`        |
| `tip`           |`"admonition is-success"`        |
| `custom`        |`"admonition is-category-custom"`|

All these admonitions can be used in the following way

```markdown
!!! tip "Header"
    Text ...
    ```@repl
    a = 1
    b = 2
    a + b
    ```
```

The resulting admonition looks as follows

!!! tip "Header"
    Text ...
    ```@repl
    a = 1
    b = 2
    a + b
    ```

The problem is, that the evaluation of the block of code inside admonitions is not currently supported. To allow code evaluation inside admonitions, we can use two raw HTML blocks to wrap the admonition body. The syntax of the code between the HTML block is the same as everywhere else in the document.

````html
```@raw html
<div class="admonition is-success">
<header class="admonition-header">Header</header>
<div class="admonition-body">
```

Text ...

```@repl
a = 1
b = 2
a + b
```

```@raw html
</div></div>
```
````

The resulting admonition then looks as follows

```@raw html
<div class="admonition is-success">
<header class="admonition-header">Header</header>
<div class="admonition-body">
```

Text ...

```@repl
a = 1
b = 2
a + b
```

```@raw html
</div></div>
```

## Additional admonition types

The css style downloaded in the `make.jl` provides three additional admonition types

| Admonition type | html class                        |
| :--             | :--                               |
| `theorem`       |`"admonition is-category-theorem"` |
| `exercise`      |`"admonition is-category-exercise"`|
| `homework`      |`"admonition is-category-homework"`|

Currently, the first two types use the style of the default admonitions types. However, it may change in the future.

!!! theorem "Theorem"
    Text...

!!! exercise "Exercise"
    Text...

!!! homework "Homework"
    Text...

## Exercise with solution

The used css style also provides style for collapsible admonition that can be used for example to define exercise with the hidden solution. To use this feature, we have to use raw HTML blocks.

````html
```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Header</header>
<div class="admonition-body">
```

Some text that describes the exercise

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

Solution

```@raw html
</p></details>
```
````

The result is following

```@raw html
<div class="admonition is-category-exercise">
<header class="admonition-header">Header</header>
<div class="admonition-body">
```

Some text that describes the exercise

```@raw html
</div></div>
<details class = "solution-body">
<summary class = "solution-header">Solution:</summary><p>
```

Solution

```@raw html
</p></details>
```