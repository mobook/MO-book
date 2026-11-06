"""Emit identical inline JavaScript registrations only once per HTML page.

Sphinx 7.4 can retain the same inline registration with filenames None and
an empty string. External assets and distinct inline scripts are preserved.
"""

from collections.abc import Mapping


def deduplicate_inline_scripts(app, pagename, templatename, context, doctree):
    scripts = context.get("script_files")
    if scripts is None:
        return
    seen = []
    unique = []
    for script in scripts:
        attributes = getattr(script, "attributes", {})
        if (
            not getattr(script, "filename", None)
            and isinstance(attributes, Mapping)
            and "body" in attributes
        ):
            # Attribute values may contain lists or dictionaries. Compare them
            # directly rather than requiring every value to be hashable.
            key = (getattr(script, "priority", None), dict(attributes))
            if key in seen:
                continue
            seen.append(key)
        unique.append(script)
    # Keep references held by the builder and other extensions valid.
    scripts[:] = unique


def setup(app):
    app.connect("html-page-context", deduplicate_inline_scripts, priority=999)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
