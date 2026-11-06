"""Regression checks for inline-script preservation during the HTML build."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


SPEC = importlib.util.spec_from_file_location(
    "deduplicate_inline_scripts",
    Path(__file__).resolve().parents[1] / "_ext" / "deduplicate_inline_scripts.py",
)
EXTENSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXTENSION)


def script(filename=None, priority=500, **attributes):
    return SimpleNamespace(
        filename=filename, priority=priority, attributes=attributes
    )


class InlineScriptTests(unittest.TestCase):
    def deduplicate(self, scripts):
        context = {"script_files": scripts}
        EXTENSION.deduplicate_inline_scripts(None, "page", None, context, None)
        return context

    def test_later_registration_through_existing_list_is_preserved(self):
        first = script(body="initialize();")
        scripts = [first, script(filename="", body="initialize();")]
        context = self.deduplicate(scripts)
        later = script(filename="later-extension.js")
        scripts.append(later)
        self.assertIs(context["script_files"], scripts)
        self.assertEqual(context["script_files"], [first, later])

    def test_nested_attributes_do_not_need_to_be_hashable(self):
        first = script(body="initialize();", options={"items": [1, 2]})
        duplicate = script(
            filename="", body="initialize();", options={"items": [1, 2]}
        )
        different = script(body="initialize();", options={"items": [1, 3]})
        context = self.deduplicate([first, duplicate, different])
        self.assertEqual(context["script_files"], [first, different])

    def test_external_assets_and_distinct_registrations_keep_their_order(self):
        scripts = [
            script(filename="asset.js"),
            script(filename="asset.js"),
            script(body="first();"),
            script(body="second();"),
            script(priority=900, body="first();"),
            script(body="first();", defer="defer"),
        ]
        context = self.deduplicate(scripts.copy())
        self.assertEqual(context["script_files"], scripts)

    def test_context_without_scripts_is_unchanged(self):
        context = {"title": "No scripts"}
        EXTENSION.deduplicate_inline_scripts(None, "page", None, context, None)
        self.assertEqual(context, {"title": "No scripts"})


if __name__ == "__main__":
    unittest.main()
