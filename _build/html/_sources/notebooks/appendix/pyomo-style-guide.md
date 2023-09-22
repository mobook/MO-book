# Pyomo style guide

This style guide supports the development and deployment of consistent, readable, and maintainable Pyomo code for modeling and optimization. These guidelines supplement standard Python style guides conventions, such as [PEP 8](https://www.python.org/dev/peps/pep-0008/), with specific recommendations for Pyomo. Comments and suggestions are welcome.

## Workflows

A typical development workflow for Pyomo applications comprises:

* Collection and pre-processing of representative application data.
* Pyomo model development
* Computing a solution
* Post-processing and analysis of solution data
* Model testing and validation

Subsequent deployment of Pyomo model will omit the development and validation steps, but may integrate the remaining elements into existing application workflows. This style guide supports development and deployment workflows by emphasizing modularity and clean interfaces between successive stages in the standard Pyomo workflow. 

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

Pyomo model objects created with `pyo.Param`, `pyo.Var`, and `pyo.Constraint` can be indexed by elements from a Pyomo Set or RangeSet. 

Here are various ways to create a Pyomo Set:
```python
model = pyo.ConcreteModel()

# Using Python Set
model.A = pyo.Set(initialize={1, 2, 3})

# Using Python List
model.B = pyo.Set(initialize=[1, 2, 3])

# Using Python Dictionary Keys
data = {1: 'a', 2: 'b', 3: 'c'}
model.C = pyo.Set(initialize=data.keys())

# Using Python Generators
gen = (i for i in range(1, 4))
model.D = pyo.Set(initialize=gen)
```
Here are various ways to create a Pyomo RangeSet:
```python
model = pyo.ConcreteModel()

# Basic RangeSet: Creates a set of integers from 1 to 5, inclusively
model.R1 = pyo.RangeSet(1, 5)

# Basic RangeSet with only upper bound: Creates a set of integers from 1 to 5, inclusively. This is similar to Python's range function but is 1-based rather than 0-based.
model.R1bis = pyo.RangeSet(5)

# RangeSet with Step: Creates a set {1, 3, 5}
model.R2 = pyo.RangeSet(1, 5, 2)

# Reversed RangeSet: Creates a set {5, 4, 3, 2, 1}
model.R3 = pyo.RangeSet(5, 1, -1)

# Floating-point RangeSet: Creates a set {1.0, 1.5, 2.0, 2.5}
model.R4 = pyo.RangeSet(1, 2.5, 0.5)

# RangeSet with Conditional Filtering: Creates a set {1, 4}
def filter_rule(model, i):
    return i % 3 == 1

model.R5 = pyo.RangeSet(1, 5, filter=filter_rule)
```

If we use a dictionary, only the keys of the dictionary are used to initialize the set. This could be useful for cases where you have a dictionary of parameters or data, and you want a set containing the relevant keys. Generators can be particularly useful when you have a large set that can be generated using some logic or function, without needing to store all elements in memory.

Alternatively, Pyomo model objects can be indexed with iterable Python objects such as sets, lists, dictionaries, and generators. Indexing with a Pyomo Set or RangeSet is preferred for most circumstances. There are several reasons why:

* Consistent use of Pyomo Set and RangeSet enhances readability by providing a consistent expression of models.

* Pyomo uses Sets and RangeSets to trace model dependencies which provides better error reporting and sensitivity calculations.

* Pyomo Set and RangeSet provides a clear and uniform separation between data pre-processing and model creation. 

  ```python
  # Data Pre-Processing
  time_periods = [1, 2, 3]
  locations = ['NY', 'SF', 'LA']

  # Model creation
  model = pyo.ConcreteModel()

  # Create sets based on pre-processed data
  model.T = pyo.Set(initialize=time_periods)
  model.L = pyo.Set(initialize=locations)

  # Define cariables indexed by Sets
  model.x = pyo.Var(model.T, model.L)
  ```

* Pyomo Set provides additional features for  model building and deployment, including filtering and data validation.

  ```python
  # Filtering with Pyomo Set
  model.I = pyo.Set(initialize=[1, 2, 3, 4], filter=lambda model, i: i % 2 == 0)
  ```

* Pyomo creates an associated internal Pyomo Set each time Python iterables are used to create indexed model objects.  Creation of multiple objects with the same iterable results in redundant internal sets.

  ```python 
  # Less efficient due to redundant internal sets
  model.x = pyo.Var([1, 2, 3])
  model.y = pyo.Var([1, 2, 3])

  # More efficient with Pyomo Set
  model.I = pyo.Set(initialize=[1, 2, 3])
  model.x = pyo.Var(model.I)
  model.y = pyo.Var(model.I)
  ```

Furthermore, given a Python dictionary that specifies the bounds on decision variables

```python
bounds = {"a": 12, "b": 23, "c": 14}
```

the following

```python
m = pyo.ConcreteModel()
m.B = pyo.Set(initialize=list(bounds.keys()))
m.x = pyo.Var(m.B)
```

is preferred to

```
m = pyo.ConcreteModel()
m.x = pyo.Var(bounds.keys())
```

In many optimization problems, we often need to construct a specific set of constraints for a subset of an existing set (for example to exclude the boundary items). Rather than creating a new set

```python
model.U = pyo.Set([v for v in model.V if v not in U]) 

@model.Constraint(model.U)
def constraint(model, v):
  return expression(v) == 1
```
it is preferable to keep the number of sets as small as possible and add conditionals in the constraint rule together with `pyo.Constraint.Skip` as follows

```python
@model.Constraint(model.V)
def constraint(model, v):
  if v in U:
     return expression(v) == 1
  else:
     return pyo.Constraint.Skip
```

### Parameters

[Pyomo parameters](https://pyomo.readthedocs.io/en/stable/pyomo_modeling_components/Parameters.html)  are created with the `pyo.Param()` class. They are used to localize parameter values to a specific model or block. Parameters can be indexed by set, initialized, restricted to specific domains, and include callbacks for validation.

```python
model = pyo.ConcreteModel()

# Single parameter initialized with a single value
model.p1 = pyo.Param(initialize=10)

# Parameters initialized using a dictionary
parameters = {1: 10, 2: 20, 3: 30}
model.I = pyo.Set(initialize=parameters.keys())
model.p2 = pyo.Param(model.I, initialize=parameters)

# Parameter with validation callbacks to check the validity of parameter values
def validate_param(model, value, index):
    return value >= 0 and value <= 100

model.p3 = Param(model.I, validate=validate_param, initialize={1: 10, 2: 20, 3: 30})
```

Given a sets `model.I` and `model.J`, parameters indexed as `model.a[i, j]` can be defined by

```python
model = pyo.ConcreteModel()

model.I = pyo.RangeSet(5)
model.J = pyo.RangeSet(3)

@model.Param(model.I, model.J, domain=pyo.NonNegativeReals)
def a(model, i, j):
    return i**2 + j**2
```

where the function `a` returns a non-negative numeric value.

By default, Pyomo parameters are immutable which assures their values will be consistent throughout the model construction and transformations. Parameters determining the size of index sets, or fixed upper and lower bounds on decision variables, are examples where using an immutable Pyomo parameter is good practice. 

```python
def build_model(a, b):
    # Use 'a' and 'b' as parameters within this function
    model = pyo.ConcreteModel()
    model.param_a = pyo.Param(initialize=a)
    model.param_b = pyo.Param(initialize=b)
    # ...
    return model
```

Pyomo parameters created with `mutable=True` are used to build models that can be re-solved for parametric or sensitivity analysis. The use of mutable parameters should be limited and intentional.

Pyomo modelers often prefer to use native Python data structures. In these cases, best practice is to limit the scope of the parameters by constructing the model within a Python function, and using function arguments of the function to provide a clear interface to the global scope.

Consistent with good programming practice, global non-constant parameters should be avoided. Non-constant global parameters lead to inconsistent state in the code and complicate understanding of models when changed or redefined elsewhere in the code.

### Variables

#### Use `domain` rather than `within` 

The `pyo.Var()` class accepts either `within` or `domain` as a keyword to specify decision variables. Offering options with no functional difference places an unnecessary cognitive burden on new users.   Consistent use of `domain` is preferred because of its common use in mathematics to represent the set of all values for which a variable is defined.

```python
model.x = pyo.Var(domain=pyo.NonNegativeReals)
```

#### Use `bounds` when known and fixed

A Pyomo model can place bounds on decision variables with either the `bounds`  keyword in the argument to `pyo.Var`, or as explicit constraints in the model. 

When upper or lower bounds for a variable are known and fixed, use of `bounds` when creating the variable is a best practice in mathematical optimization.  This practice can reduce the number of explicit constraints in the model and simplify coding and model display. 

```python
# Using bounds (preferred)
model.x = pyo.Var(bounds=(0, 10))

# Equivalent explicit constraints (less preferred)
model.x = pyo.Var()
model.lower_bound = pyo.Constraint(expr=model.x >= 0)
model.upper_bound = pyo.Constraint(expr=model.x <= 10)
```

If, however, variable bounds may be subject to change during the course of problem solving, then explicit constraints should be used.

```python 
model.x = pyo.Var()
model.LB = pyo.Param(mutable=True, initialize=0)
model.UB = pyo.Param(mutable=True, initialize=10)

model.lower_bound = pyo.Constraint(expr=model.x >= model.LB)
model.upper_bound = pyo.Constraint(expr=model.x <= model.UB)
```
### Constraints and Objective

#### Prefer `pyo.Constraint` to `pyo.ConstraintList`

The `pyo.ConstraintList()` class is useful for creating a collection of constraints that are not naturally indexed by a set, or when the constraints are generated dynamically within a loop. 

However, to make the code more readable, maintainable, and facilitate better model formulation, `pyo.ConstraintList()` should not be used as a substitute for the more structured and readable use of `pyo.Constraint()` in combination with `pyo.Set` and decorators.

#### Use decorators to improve readability

Indexed Pyomo constraints are constructed by a rule.  When using `pyo.Constraint()`, rules are normally named by adding `_rule` as a suffix to the name of the associated constraint. For example, assuming model `m`  and the associated sets, parameters, and variables have been previously defined, 

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

The use of decorators improves readability by eliminating the need for the `rule` keyword and the need for writing multiple versions of the same constraint name. 

The decorator syntax is straightforward for objectives and simple constraints. For Python users unfamiliar with decorators, decorators can be described as a way to 'tag' functions that are to be incorporated into the Pyomo model. Indices and keywords are used modify the extend to the bahavior of the decorator. 

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
  return sum(model.ship[src, dst] for dst in model.DESTINATIONS) <= model.capacity[src]

@model.Constraint(model.DESTINATIONS)
def demand_constraint(model, dst):
  return sum(model.ship[src, dst] for dst in model.SOURCES) <= model.demand[dst]
```

## Naming conventions

The choice of constraint and variables names is important for readable Pyomo models. Good practice is to use descriptive lower case names with words separated by underscores consistent with [PEP 8](https://peps.python.org/pep-0008/) recommendations. 

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

Parameter names, especially mutable parameters intended for use in parametric studies, may use capitalized words (i.e., "CamelCase").

### Use descriptive Constraint and Variable names

Objectives, constraints, variables, disjuncts, and disjunctions should use descriptive names following PEP 8 guidelines with lower case words separated by underscore (i.e, "snake_case").  

As an exception for small tutorial examples where mathematical formulation accompanies the model,  the corresponding Pyomo model may use the same variable and parameter name. For example, a mathematical model written as

$$
\begin{aligned}
& & f = \max_{x,  y}\quad & 40x + 30y\\
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

This practice is generally discouraged, because the resulting models are not easily read without reference to the accompanying mathematical notes. Pyomo includes a `.doc`  attribute that can be used to document relationships between the Pyomo model and any reference materials.

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

## Data styles and conventions

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

## Multi-dimensional or multi-indexed data

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





