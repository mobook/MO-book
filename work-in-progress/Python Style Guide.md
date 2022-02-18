# Pyomo Style Guide

The goal of this style guide is to standardize the development, presentation, and deployment of Pyomo models.

## Pyomo Coding

### Use `pyo` for Pyomo namespace

For consistency with Pyomo documentation and the Pyomo book, the preferred namespace convention for Pyomo is `pyo` 

```python
import pyomo.environ as pyo
```

### Use `ConcreteModel`  rather than `AbstractModel`

Instances of Pyomo optimization models can be create with `AbstractModel` or `ConcreteModel` as described in the [documentation](https://pyomo.readthedocs.io/en/stable/pyomo_overview/abstract_concrete.html). `AbstractModel` specifies a model with symbolic parameters which can be specified later to define a specific model instance. A `ConcreteModel` requires parameter values to be specified when the model is specified. 

Because Pyomo is embedded within Python, the preferred to method for creating model instances is a Python function or class that accepts parameter values and returns a `ConcreteModel` model instance.

### Shorten model and block names when possible

For brevity and clarity, model and block names should be short and consistent with PEP 8 naming standards. A single lower case `m` is acceptable in most instances of a model with a single block. 

### Use Pyomo Sets for indexing

Pyomo  modeling elements including `Param`,`Var`, `Constraint` should be indexed by Pyomo Sets rather than utterable Python objects.  For example, given a Python dictionary

```
bounds = {"a": 12, "b": 23, "c": 14}
```

The following

```
m.B = pyo.Set(initialize=bounds.keys())
m.x = pyo.Var(m.B)
```

Is preferred, rather than

```
m.x = pyo.Var(boounds.keys())
```

Consistent with conventional mathematical notation in optimization, use of upper-case letters to denote sets is an acceptable deviation from PEP style guidelines.

## Working with Data

### Use Tidy Data

Tidy data is a semantic model for of organizing data sets. The core principle of Tidy data is that each data set is organized by rows and columns where each entry is a single value. Each column contains all data associated with single variable. Each row contains all values for a single observation. 

Tidy data may be read in multiple ways. Pandas `DataFrame` objects are particularly well suited to Tidy Data and recommended for reading, writing, visualizing, and displaying Tidy Data.

When using doubly nested Python dictionaries for Tidy Data, the primary keys will provide unique identifiers for each observation. Each observation is a dictionary where  secondary keys label the variables and each entry will consist of a single value.

Alternative structures may include nested lists, lists of dictionaries, or numpy arrays. In each case a single data will be referenced as `data[obs][var]` where `obs` identifies a particular observation or slice of observations, and `var` identifies a variable.  

Higher dimensional data structures should encode values as `data[ds][obs][var]` where `ds` identifies a data set composed of Tidy Data.

### Use Pandas for display and visualization

The Pandas library provides an extensive array of functions for the manipulation, display, and visualization of data sets. 







