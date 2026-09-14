# Hands-On Mathematical Optimization with Python

Welcome to this repository of companion notebooks for the book _[Hands-On Mathematical Optimization with Python](https://www.cambridge.org/highereducation/books/hands-on-mathematical-optimization-with-python/F15ABA8AF886E7E6F7444151F40683A1#overview)_, published by Cambridge University Press.

The book can be purchased on [the CUP website](https://www.cambridge.org/us/universitypress/subjects/mathematics/optimization-or-and-risk-analysis/hands-mathematical-optimization-python?format=PB), on [Amazon](https://www.amazon.com/Hands-Mathematical-Optimization-Python-Krzysztof/dp/1009493507), and many other stores. If you are a lecturer interested in adopting this book for your course, you can request an inspection copy [filling this form](https://cup.my.salesforce-sites.com/Samples?isbn=9781009493505&Title=Hands-On+Mathematical+Optimization+with+Python&Author=Postek+et+al).

This book introduces the concepts and tools of mathematical optimization with examples from a range of disciplines. The goals of these companion notebooks are to:

- Provide a foundation for hands-on learning of mathematical optimization,
- Demonstrate the tools and concepts of optimization with practical examples,
- Help readers to develop the practical skills needed to build models and solving problem using state-of-the-art modeling languages and solvers.

## Getting started

These notebooks use **Pyomo** to describe optimization models and a **solver** to compute their solutions. HiGHS is the open-source solver used in most linear and mixed-integer linear examples.

You can work locally in Jupyter or open a notebook in Google Colab using the rocket icon at the top of its page. To install Pyomo and HiGHS from a notebook code cell, run:

```ipython
%pip install pyomo highspy
```

`%pip` installs into the current notebook kernel's environment. From a terminal, use `python -m pip install pyomo highspy` instead.

Follow each notebook's preamble for additional packages or solvers. The runnable [Solvers used in this book](notebooks/appendix/installing-pyomo-and-solvers.ipynb) appendix provides a quick installation check, explains solver choices and Colab setup, and covers open-source license terms, free commercial editions and academic access.

Start your journey with the [first chapter](notebooks/01/01.00.md)!

## Help us

We seek your feedback! If you encounter an issue or have suggestions on how to make these examples better, please open an issue using the link at the top of every page (look for the Github cat icon).

## About us

We are a group of researchers and educators who came together with a common purpose of developing materials for use in our classroom teaching. Hopefully, these materials will find use in other classrooms and, most importantly, by those seeking entry into the world of building optimization models for data-rich applications.

- Krzysztof Postek, Boston Consulting Group (formerly TU Delft)
- Alessandro Zocca, VU Amsterdam
- Joaquim Gromicho, University of Amsterdam
- Jeffrey Kantor, University of Notre Dame

## Citation

If you wish to cite this work, please use

```
@book{PZGK2025book,
  author = {Postek, Krzysztof and Zocca, Alessandro and Gromicho, Joaquim and Kantor, Jeffrey},
  title = {{Hands-On Mathematical Optimization with Python}},
  year = {2025},
  publisher = {Cambridge University Press},
  place={Cambridge},
  doi={10.1017/9781009493512}
}
```

and

```
@online{PZGK2025online,
  author = {Postek, Krzysztof and Zocca, Alessandro and Gromicho, Joaquim and Kantor, Jeffrey},
  title = {Companion Jupyter Book for {``Hands-On Mathematical Optimization with Python’’}},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/mobook/MO-book}},
}
```
