"""Emit identical inline JavaScript registrations only once per HTML page.

Sphinx 7.4 can retain the same inline registration with filenames None and
an empty string. External assets and distinct inline scripts are preserved.
"""


def deduplicate_inline_scripts(app, pagename, templatename, context, doctree):
    seen = set()
    scripts = []
    for script in context.get("script_files", []):
        attributes = getattr(script, "attributes", {})
        if not getattr(script, "filename", None) and "body" in attributes:
            key = (script.priority, tuple(sorted(attributes.items())))
            if key in seen:
                continue
            seen.add(key)
        scripts.append(script)
    context["script_files"] = scripts


def setup(app):
    app.connect("html-page-context", deduplicate_inline_scripts, priority=800)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
