# Companion Jupyter notebooks for *Hands-On Mathematical Optimization with Python*

This is the source repository for the [collection of Jupyter notebooks](https://mobook.github.io/MO-book/) associated with the book *Hands-On Mathematical Optimization with Python* published by Cambridge University Press (2025).

The book is available from [this webpage](https://www.cambridge.org/us/universitypress/subjects/mathematics/optimization-or-and-risk-analysis/hands-mathematical-optimization-python?format=PB) and [Amazon](https://www.amazon.com/Hands-Mathematical-Optimization-Python-Krzysztof/dp/1009493507). If you are a lecturer interested in adopting this book for your course, you can request an inspection copy by [filling out this form](https://cup.my.salesforce-sites.com/Samples?isbn=9781009493505&Title=Hands-On+Mathematical+Optimization+with+Python&Author=Postek+et+al).

## Citation

Please cite the book when using these companion notebooks:

```bibtex
@book{PZGK2025book,
  author = {Postek, Krzysztof and
            Zocca, Alessandro and
            Gromicho, Joaquim A. S. and
            Kantor, Jeffrey C.},
  title = {{Hands-On Mathematical Optimization with Python}},
  year = {2025},
  publisher = {Cambridge University Press},
  address = {Cambridge},
  doi = {10.1017/9781009493512}
}
```

To identify the online companion materials separately, use:

```bibtex
@misc{PZGK2025online,
  author = {Postek, Krzysztof and
            Zocca, Alessandro and
            Gromicho, Joaquim A. S. and
            Kantor, Jeffrey C.},
  title = {Companion Jupyter notebooks for
           {Hands-On Mathematical Optimization with Python}},
  year = {2025},
  howpublished = {\url{https://mobook.github.io/MO-book/}},
  note = {Source repository: \url{https://github.com/mobook/MO-book}}
}
```

Machine-readable citation information, with the book as the preferred citation, is available in [CITATION.cff](https://github.com/mobook/MO-book/blob/main/CITATION.cff).

## Building the companion website

From the repository root, install the website dependencies and build the book:

```sh
python -m pip install -r requirements.txt
jupyter-book build .
```

Open `_build/html/index.html` to preview the result. The build publishes the notebooks' saved outputs; it does not execute their code. To run a notebook, follow its preamble and the [solver overview](https://mobook.github.io/MO-book/notebooks/appendix/installing-pyomo-and-solvers.html) for the additional modeling packages and solvers.

The companion code is distributed under the [MIT License](LICENSE).

For heading and notebook conventions, see [CONTRIBUTING.md](CONTRIBUTING.md).
