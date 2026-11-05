"""Map notebook cell-ID links to their canonical MyST heading targets."""

from html.parser import HTMLParser

from docutils import nodes


class CellAnchors(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = set()

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attrs = dict(attrs)
            cell_id = attrs.get("data-notebook-cell-id")
            target = attrs.get("name") or attrs.get("id")
            if cell_id and target:
                self.targets.add(("cell-id=" + cell_id, target))


def register_cell_anchors(app, doctree):
    slugs = app.env.metadata[app.env.docname].setdefault("myst_slugs", {})
    for node in doctree.findall(nodes.raw):
        if node.get("format") != "html":
            continue
        parser = CellAnchors()
        parser.feed(node.astext())
        for cell_link, target in parser.targets:
            if target in slugs:
                slugs.setdefault(cell_link, slugs[target])


def setup(app):
    app.connect("doctree-read", register_cell_anchors)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
