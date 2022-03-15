# Pyomo Style Guide

This style guide supports the development and deployment of consistent, readable, and maintainable Pyomo code for modeling and optimization. These guidelines supplement standard Python style guides conventions, such as [PEP 8](https://www.python.org/dev/peps/pep-0008/), with specific recommendations for Pyomo. Comments and suggestions are welcome.

## Workflows

A typical development workflow for Pyomo applications comprises:

* Collection and pre-processing of representative application data.
* Pyomo model development
* Computing a solution
* Post-processing and analysis of solution data
* Model testing and validation

Subsequent deployment of Pyomo model will omit the development and validation steps, but may integrate the remaining elements into existing application workflows. This style guide supports development and deployment workflows by emphasizing modularity and clean interfaces between successive steps. 

## Coding Conventions

### Use `pyo` for the Pyomo namespace

For consistency with Pyomo [documentation](https://pyomo.readthedocs.io/en/stable/pyomo_overview/abstract_concrete.html) and the Pyomo book, the preferred namespace convention for Pyomo is `pyo` 

```python
import pyomo.environ as pyo
```

The usage

```python
# don't do this
from pyomo.environ import *
```

is strongly discouraged. For special cases where a less verbose style is needed, such as presentations or introducing Pyomo to new users, a better practice is to explicitly import the needed Pyomo objects as shown in this example:

```python
# for presentations or teaching examples
from pyomo.environ import ConcreteModel, Var, Objective, maximize, SolverFactory
```

### Use `pyo.ConcreteModel`  instead of `pyo.AbstractModel`

The preferred method for creating instances of Pyomo models is a Python function or class that accepts parameter values and returns a `pyo.ConcreteModel`.

Pyomo provides two methods for creating model instances, `pyo.AbstractModel` or `pyo.ConcreteModel`.  A `pyo.ConcreteModel` requires parameter values to be known when the model is created. `pyo.AbstractModel` specifies a model with symbolic parameters which can be specified later to define an instance of the  model. However, because Pyomo is integrated within Python,  `pyo.ConcreteModel` model instances can be created with a Python function or class using the full range of language features. For this reason, there is  little benefit for  `pyo.AbstractModel`.

### Index with Pyomo Set and RangeSet

Pyomo model objects created with `pyo.Param`, `pyo.Var`, and `pyo.Constraint` can be indexed by elements from a Pyomo Set or RangeSet. Alternatively, Pyomo model objects can be indexed with iterable Python objects such as sets, lists, dictionaries, and generators.

Indexing with a Pyomo Set or RangeSet is preferred for most circumstances. There are several reasons why:

* Pyomo Set and RangeSet provides a clear and uniform separation between data pre-processing and model creation. 
* Consistent use of Pyomo Set and RangeSet enhances readability by providing a  consistent expression of models.
* Pyomo Set provides additional features for  model building and deployment, including filtering and data validation.
* Pyomo uses Sets and RangeSets to trace model dependencies which provides better error reporting and sensitivity calculations.
* Pyomo creates an associated internal Pyomo Set each time Python iterables are used to create indexed model objects.  Creation of multiple objects with the same iterable results in redundant internal sets.

 Given a Python dictionary

```python
bounds = {"a": 12, "b": 23, "c": 14}
```

the following

```python
m = pyo.ConcreteModel()
m.B = pyo.Set(initialize=bounds.keys())
m.x = pyo.Var(m.B)
```

is preferred to

```
m = pyo.ConcreteModel()
m.x = pyo.Var(bounds.keys())
```

### Parameters

Pyomo modelers may prefer to use native Python data structures rather declare and use instances of parameters created using the `pyo.Param()` class.  Use of Pyomo parameters, however, is encouraged for  particular circumstances.  

By default, Pyomo parameters are immutable which can prevent inadvertent changes to key model parameters. Parameters that define the size of index sets, or establish fixed upper or lower bounds on variables, are examples where defining an immutable Pyomo parameter is good practice.

Pyomo parameters should also be used 

* with `mutable=True`  when a model will be solved for multiple parameter values. 
* when the use of native Python data structures would reduce readability.
* when developing complex model requiring clear interfaces among modules that document model data, provide default values and validation.

### Variables

#### Use `domain` rather than `within` 

The `pyo.Var()` class accepts either `within` or `domain` as a keyword to specify decision variables. Offering options with no functional difference places an unnecessary cognitive burden on new users.   Consistent use of `domain` is preferred because of its common use in mathematics to represent the set of all values for which a variable is defined.

#### Use `bounds` when known and fixed

A Pyomo model can place bounds on decision variables with either the `bounds`  keyword in the argument to `pyo.Var`, or as explicit constraints in the model. 

When upper or lower bounds for a variable are known and fixed, use of `bounds` when creating the variable is a best practice in mathematical optimization.  This practice can reduce the number of explicit constraints in the model and simplify coding and model display. 

If, however, variable bounds may be subject to change during the course of problem solving, then explicit constraints should be used.

### Constraints and Objective

#### Prefer `pyo.Constraint` to `pyo.ConstraintList`

The `pyo.ConstraintList()` class is useful for creating a collection of constraints for which there is no simple indexing,  such as implementing algorithms featuring constraint generation. However, ConstraintList should not be used as a substitute for the more structured and readable use of `pyo.Constraint()`. 

#### Use decorators to improve readability

Indexed Pyomo constraints are constructed by a rule.  When using `pyo.Constraint()`  rules are normally named by adding `_rule` as a suffix to the name of the associated constraint. For example, assuming model `m`  and the associated sets, parameters, and variables have been previously defined, 

```python
def new_constraint_rule(m, s):
  return m.x[s] <= m.ub[s]
m.new_constraint = pyo.Constraint(m.S, rule=new_constraint_rule)
```

A recent innovation in Pyomo is the use of Python decorators to create Constraint, Objective, and Disjunction objects. Using decorators, the above example is written as

```python
@m.Constraint(m.S)
def new_constraint_rule(m, s):
  return m.x[s] <= m.ub[s]
```

The use of decorators improves readability by eliminating the need for the `rule` keyword and writing multiple versions of the constraint name. 

The decorator syntax is straightforward for objectives and simple constraints. Keywords are included in the decorator. 

````python
@model.Constraint()
def demand_constraint(model):
  return model.x + model.y <= 40

@model.Objective(sense=pyo.maximize)
def profit(model):
  return 3*model.x + 4*model.y
````

Indices are also included in the decorator for indexed objects.

```python
@model.Constraint(model.SOURCES)
def capacity_constraint(model, src):
  return sum(model.ship[src, dst] for dst in model.DESTINATIONS) <= model.CAPACITY[src]

@model.Constraint(model.DESTINATIONS)
def demand_constraint(model, dst):
  return sum(model.ship[src, dst] for dst in model.SOURCES) <= model.DEMAND[dst]
```

## Naming Conventions

The choice of constraint and variables names is important for readable Pyomo models. Good practice is to use descriptive lower case names with words separated by underscores consistent with PEP 8 recommendations. 

Pyomo models commonly use alternative conventions to enhance readability by visually distinguishing components of a model.

### Prefer short model and block names

Model and block names should be consistent with PEP 8 naming standards (i.e., all lowercase with words separated by underscore). Short model names are preferred for readability and to avoid excessively long lines. A single lower case `m` is acceptable in instances of a model with a single block. 

Complex models may require more descriptive names for readability. 

### Set and RangeSet names may be all caps

Consistent with common mathematical conventions in optimization modeling, use of upper-case names to denote Pyomo sets is an acceptable deviation from PEP style guidelines. Corresponding lower case name can then be used to denote elements of the set. For example, the objective
$$
\tau^\text{total} = \min \sum_{\text{machine} \in \text{MACHINES}} \tau^\text{finish}_\text{machine}
$$
may be implemented as

```python
import pyomo.environ as pyo

m = pyo.ConcreteModel()
m.MACHINES = pyo.Set(initialize=["A", "B", "C"])
m.finish_time = pyo.Var(m.MACHINES, domain=pyo.NonNegativeReals)

@m.Objective(sense=pyo.minimize)
def total_time(m):
  return sum(m.finish_time[machine] for machine in m.MACHINES)
```

### Parameter names may be capitalized

JCK: ADD NOTES ON LIMITING SCOPE OF PARAMETERS

Parameter names, especially mutable parameters intended for use in parametric studies, may use capitalized words (i.e., "CamelCase").

### Use descriptive Constraint and Variable names

Objectives, constraints, variables, disjuncts, and disjunctions should use descriptive names following PEP 8 guidelines with lower case words separated by underscore (i.e, "snake_case").  

As an exception for small tutorial examples where mathematical formulation accompanies the model,  the corresponding Pyomo model may use the same variable and parameter name. For example, a mathematical model written as

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

may be encoded as

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

This practice is generally discouraged. However, because the resulting models are not easily read without reference to the accompanying mathematical notes. Pyomo includes a `.doc`  attribute that can be used to document relationships between the Pyomo model and any reference materials.

```python
import pyomo.environ as pyo

# create model instance
m = pyo.ConcreteModel()

# decision variables
m.production_x = pyo.Var(domain=pyo.NonNegativeReals, doc="x")
m.production_y = pyo.Var(domain=pyo.NonNegativeReals, doc="y")

# objective
m.profit = pyo.Objective(expr = 40*m.production_x + 30*m.production_y, sense=pyo.maximize)
m.profit.doc = "f"

# declare constraints
m.labor_a = pyo.Constraint(expr = 2*m.production_x + m.production_y <= 10, doc="A")
m.labor_b = pyo.Constraint(expr = m.production_x + 2*m.production_y <= 15, doc="B")

m.pprint()

```


## Data Styles and Conventions

Reading, manipulating, and writing data sets often consumes a considerable amount of time and coding in routine projects. Standardizing on a basic set of principles for organizing data can streamline coding and model development. Below we promote the use of [Tidy Data](https://vita.had.co.nz/papers/tidy-data.html) for managing data sets associated with Pyomo models.

### Use Tidy Data

Tidy data is a semantic model for of organizing data sets. The core principle of Tidy data is that each data set is organized by rows and columns where each entry is a single value. Each column contains all data associated with single variable. Each row contains all values for a single observation. 

| scenario | demand | Price |
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

This document is the result of interactions with students and colleagues over several years. Several individuals reviewed and provided feedback on early drafts and are acknowledged here.

* David Woodruff, UC Davis
* Javier Salmeron-Medrano, Naval Postgraduate School
* Bethany Nicholson, John Siirola, Michael Bynum, and the Pyomo development team
* Jasper M. H. van Doorn
* Leon Lan





