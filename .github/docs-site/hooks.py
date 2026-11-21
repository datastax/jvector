"""MkDocs hook for the JVector docs site.

Pages in docs/ link to repository files outside docs/ (examples, YAML configs,
UPGRADING.md) with relative paths, which work on GitHub. The site only contains
docs/, so this hook rewrites those links to the same files on GitHub.
"""

import os
import posixpath
import re

REPO_URL = "https://github.com/datastax/jvector"
BRANCH = "main"

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LINK = re.compile(r"(\]\()([^)\s]+)(\))")


def _rewrite(page_dir, target):
    if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith(("#", "/")):
        return None
    path, sep, anchor = target.partition("#")
    repo_path = posixpath.normpath(posixpath.join("docs", page_dir, path))
    if repo_path == "docs" or repo_path.startswith("docs/") or repo_path.startswith(".."):
        return None
    kind = "tree" if os.path.isdir(os.path.join(_ROOT, repo_path)) else "blob"
    return f"{REPO_URL}/{kind}/{BRANCH}/{repo_path}{sep}{anchor}"


def on_page_markdown(markdown, page, config, files):
    page_dir = posixpath.dirname(page.file.src_uri)

    def replace(match):
        new = _rewrite(page_dir, match.group(2))
        return match.group(0) if new is None else match.group(1) + new + match.group(3)

    return _LINK.sub(replace, markdown)
