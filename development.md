# Development Notes

This project is maintained using [JupyterBook](https://jupyterbook.org/intro.html).

## Media

Video files need to be save in `_build/html/_static/media` where it can be referenced via the path `../_static/media/`

## Building

After completing routine edits to notebooks, data, or configuration files, open a terminal window and navigate to the directory of the local git repository. 

```
jupyterbuild clean ../cbe30338-book
```

Rebuild the book by executing the following command from the terminal window while 

```
jupyterbuild build ../cbe30338-book
```

Commit and push changes to the remote git repository.

```
git add --all
git commit -m "commit message"
git push
```

Move relevant html files to github pages.

```
ghp-import -n -p -f _build/html
```
