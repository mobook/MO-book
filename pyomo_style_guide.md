# Pyomo Style Guide

This style guide supports the development of consistent, readable, and maintainable Pyomo models. These guidelines supplement standard Python style guides conventions, such as PEP 8, with specific recommendations for Pyomo. Comments and suggestions are welcome.

## Organizing a Pyomo Project

An optimization calculation typically involve a sequence of discrete steps:

1. **Data Preprocessing**. Retrieve and validate application data, wrangle data in preparation for model building. Intermingling data preprocessing and model building is to be avoided. To the practical extent possible, data preprocessing should be isolated from model building through the use of semantic organization (c.f., "Tidy Data" and "Tsibble"), Python data structures to organize data (c.f., nest dictionaries and Pandas), and use of Pyomo Set. 
2. Creating a Model Instance.
3. Solving a Model Instance.
4. Post processing a model solution.

## Pyomo Rules

### Use `pyo` for the Pyomo namespace

The preferred namespace convention for Pyomo is `pyo` 

```python
import pyomo.environ as pyo
```

Use of `pyo`  provides consistency with Pyomo [documentation](https://pyomo.readthedocs.io/en/stable/pyomo_overview/abstract_concrete.html) and the Pyomo book.  The usage

```python
# don't do this
import pyomo.environ as *
```

Is strongly discouraged. In special cases where a less verbose style is desired, such as presentations or introducing Pyomo to new users, explicitly import the needed Pyomo objects. For example

```python
# for presentations or teaching examples
from pyomo.environ import ConcreteModel, Var, Objective, maximize, SolverFactory
```



### Use `ConcreteModel`  instead of `AbstractModel`

The preferred method for creating instances of Pyomo models is to use a Python function or class that accepts parameter values and returns a `ConcreteModel`.

Pyomo provides two methods for creating model instances, `AbstractModel` or `ConcreteModel`.  A `ConcreteModel` requires parameter values to be known when the model is specified. `AbstractModel` specifies a model with symbolic parameters which can be specified later to define a specific instance of the generic model. However, because Pyomo is embedded within Python,  `ConcreteModel` model instances can be created in Python function or class using the full range of language features. For this reason, there is  little practical need or benefit for  `AbstractModel`.

### Prefer short model and block names

Model and block names should be consistent with PEP 8 naming standards (i.e., all lowercase with words separated by underscore). Short model names are preferred for readability and to avoid excessively long lines. A single lower case `m` is acceptable in  instances of a model with a single block. 

Complex models may require more descriptive names for readability. 

### Indexing with Pyomo Set and RangeSet

Pyomo model objects created with  `Param`,`Var`, `Constraint` that can be indexed by elements from a Pyomo Set or from iterable Python objects such as sets, lists, and dictionaries. 

A Pyomo Set or RangeSet is preferred for most circumstances. Consistent use of Pyomo Set provides a  consistent expression of models which enhances readability. Use of Pyomo Set provides a clear interface between data "wrangling" and model creation making it easier to identify run-time bugs and refactor code. Pyomo Set also provides additional features useful  model building and deployment, including filtering and validation data.

For a Python dictionary

```python
bounds = {"a": 12, "b": 23, "c": 14}
```

the following

```python
m.B = pyo.Set(initialize=bounds.keys())
m.x = pyo.Var(m.B)
```

is preferred to

```
m.x = pyo.Var(bounds.keys())
```

For consistency with standard mathematical conventions, upper-case letters to denotes Pyomo sets is an acceptable deviation from PEP style guidelines. Lower case letters can be used to denote elements of the set. For example, the objective
$$
f = \min \sum_{b\in B} x_b
$$
may be implemented as

```python
m.f = pyo.Objective(expr=sum(m.x[b] for b in m.B))
```



### Indexing with Python iterables

Pyomo model objects may use of iterable Python objects, including Python generators, for indexing.  This may be preferred to the use of Pyomo Set and RangeSet for models indexed by large data sets. Python iterables, iterators, and generators can reduce the memory footprint and speed up model generation.  For model maintenance, care should be taken to document the rational behind these design decisions.

### Variable and Parameter names

The choice of variable and parameter names are crucial for readable Pyomo models.

When a formal mathematical formulation accompanies the documentation, Pyomo model may use the same variable and parameter name. For example, a mathematical model written as


$$
\begin{aligned}
& & f = \max_{x,  y}\quad & 3x + 4y\\
\\
& \text{subject to}
\\
& & 2x + y  & \leq 10 \\
& & 
x + 2y & \leq 15 \\
\end{aligned}
$$

May be encoded

```python
import pyomo.environ as pyo

# create model instance
m = pyo.ConcreteModel()

# decision variables
m.x = pyo.Var(domain=pyo.NonNegativeReals)
m.y = pyo.Var(domain=pyo.NonNegativeReals)

# objective
m.f = pyo.Objective(expr = 40*m.x + 30*m.y, sense=pyo.maximize)

# declare constraints
m.a = pyo.Constraint(expr = 2*m.x + m.y <= 10)
m.b = pyo.Constraint(expr = m.x + 2*m.y <= 15)

m.pprint()
```

When Pyomo models are not accompanied by mathematical documentation defining the variables, parameters, constraints, and objectives,  then standard Python naming conventions should be used. Following PEP 8, these names should be lower case with words separated by underscores to improve readability.

### Use `domain` rather than `within` 

The `pyomo.Var()` class accepts either `within` or `domain` as a keyword to specify decision variables. Offering options with no functional difference places an unnecessary cognitive burden on new users.   The use of `domain` is preferred because of its consistent use in mathematics to represent the set of all values for which a variable is defined.

### Use `bounds` in `Var` when known

Use of `bounds` is strongly encouraged as a best practice in mathematical optimization.  Providing bounds in `Var` eliminates the need for a constraint, simplifies coding and model display.

### Use lambda functions to improve readability

Indexed constraints require a rule to generate the constraint from problem data. A rule is a Python function that returns a Pyomo equality or inequality expression, or a Python 3-tuple of the form (lb, expr, ub). The rule accepts a model instance as the first argument, and one additional argument for each index used to specify the constraint.

In some cases rules are simple enough to express in a single line. For these cases a Python lambda expression may improve readability. For example, the indexed constraint

```python
def new_constraint_rule(m, s):
  return m.x[s] <= m.ub[s]
m.new_constraint = pyo.Constraint(m.S, rule=c_rule)
```

can be expressed in a single line.

```python
m.c = pyo.Constraint(m.S, rule=lambda m, s: m.x[s] <= m.ub[s])
```

Longer expressions can be broken into multi-line statements following PEP 8 guidelines for indentation.

```python
m.c = pyo.Constraint(m.S, rule=lambda m, s: 
          m.x[s] <= m.ub[s])
```

Note that lambda functions are limited to Pyomo expressions that can be expressed in a single line of code.

### Use rule naming conventions

A common Pyomo convention is to name rules by adding`_rule` as a suffix to the name of the associated constraint.

```python
 def new_constraint_rule(m, s):
  return m.x[s] <= m.ub[s]
m.new_constraint = pyo.Constraint(m.S, rule=c_rule)
```

### Prefer `Constraint` to `ConstraintList`

The `ConstraintList` object is useful for using to Python coding to create a series of related constraints for which there is no simple indexing. However, it should not be used as a substitute for the use the for the more readable use of`Constraint` and an associated rules.

## Working with Data

Reading, manipulating, and writing data sets often consumes a considerable amount of time and coding in routine projects. Standardizing on a basic set of principles for organizing data can streamline coding and model development. Below we promote the use of Tidy Data for managing data sets associated with Pyomo models.

### Use Tidy Data

Tidy data is a semantic model for of organizing data sets. The core principle of Tidy data is that each data set is organized by rows and columns where each entry is a single value. Each column contains all data associated with single variable. Each row contains all values for a single observation. 

| senario | demand | Price |
| ------- | -----: | ----: |
| high    |    200 |    20 |
| medium  |    100 |    18 |
| low     |     50 |    15 |

Tidy data may be read in multiple ways. Pandas `DataFrame` objects are well suited to Tidy Data and recommended for reading, writing, visualizing, and displaying Tidy Data.

When using doubly nested Python dictionaries, the primary keys should provide unique identifiers for each observation. Each observation is a dictionary. The secondary keys label the variables in the observation, each entry consisting of a single value.

```python
scenarios = {
  "high": {"demand": 200, "price": 20},
  "medium": {"demand": 100, "price": 18},
  "low": {"demand": 50, "price": 15}
}
```

Alternative structures may include nested lists, lists of dictionaries, or numpy arrays. In each case a single data will be referenced as `data[obs][var]` where `obs` identifies a particular observation or slice of observations, and `var` identifies a variable.  

## Multi-dimensional or Multi-indexed Data

Pyomo models frequently require $n$-dimensional data, or data with $n$ indices. Following the principles of Tidy Data, variable values should appear in a single column, with additional columns to uniquely index each value.

For example, the following table displays data showing the distance from a set of warehouses to a set of customers.

|             | Customer 1 | Customer 2 | Customer 3 |
| ----------- | ---------- | ---------- | ---------- |
| Warehouse A | 254        | 173        | 330        |
| Warehouse B | 340        | 128        | 220        |
| Warehouse C | 430        | 250        | 225        |
| Warehouse D | 135        | 180        | 375        |

The distance variable is distributed among multiple columns. Reorganizing the data using Tidy Data principles results in a table with the all values for the distance variable in a single column.

| Warehouse   | Customer   | Distance |
| ----------- | ---------- | -------- |
| Warehouse A | Customer 1 | 254      |
| Warehouse B | Customer 1 | 340      |
| Warehouse C | Customer 1 | 430      |
| Warehouse D | Customer 1 | 135      |
| Warehouse A | Customer 2 | 173      |
| Warehouse B | Customer 2 | 128      |
| Warehouse C | Customer 2 | 250      |
| Warehouse D | Customer 2 | 180      |
| Warehouse A | Customer 3 | 330      |
| Warehouse B | Customer 3 | 220      |
| Warehouse C | Customer 3 | 225      |
| Warehouse D | Customer 3 | 375      |

When working with multi-dimensional data, or other complex data structures, special care should be taken to factor "data wrangling" from model building. 

### Use Pandas for display and visualization

The Pandas library provides an extensive array of functions for the manipulation, display, and visualization of data sets. 

## Acknowledgements

This document is the result of interactions with students and colleagues over several years. Several individuals reviewed and provided feedback on early drafts. 

* David Woodruff, UC Davis

* Javier Salmeron-Medrano, Naval Postgraduate School





