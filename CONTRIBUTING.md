# Editorial conventions

Use sentence case for chapter and notebook titles: capitalize the first word and preserve proper names and acronyms such as Pyomo, HiGHS, BIM, SAA and CVXPY. After a colon introducing a subtitle, capitalize the subtitle's first word. Section headings follow the same convention as their notebooks are revised; existing section headings are being aligned progressively under issue #74.

Keep numbered notebook titles in the form `N.M Title`. Use `Extra material: Title` for unnumbered bonus notebooks. Apply the same label on the chapter landing page. Keep notebook filenames and public URLs stable. Preserve an existing public heading anchor when changing its wording.

Use one level-one heading for the notebook title, followed by level-two sections and level-three subsections. Python comments inside fenced code examples are not document headings.

Keep copyedits separate from changes to models, data, solver choices or computational results. The website builds the saved notebook outputs without executing the code. Check that headings, equations, links and figures render, and preserve code and saved outputs when a revision is purely editorial.

## Building the companion

Install the build dependencies together from `requirements.txt`, including when
updating an existing local build environment:

```console
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
jupyter-book build . --all
```

The file fixes a tested Sphinx and book-theme combination. An older theme pair
can give ScrollSpy a missing navigation target and misinterpret an external
Google Sheets fragment as a CSS selector. Update the build dependencies rather
than changing the Sheets link or editing the generated HTML. These dependencies
build the website; notebook packages and solvers follow their own preambles.
