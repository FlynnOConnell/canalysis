"""
# displayable_path.py

Module(utils/file_helpers): Class for displaying path tree.
"""
from __future__ import annotations

from pathlib import Path


class DisplayablePath:
    """
    Class for creating a tree to display current directory.

    from https://stackoverflow.com/a/49912639

    Parameters
    ----------
    path : str
       - path to innermost dir
    parent_path : str | Path
       - path to outermost dir

    Example Usage
    ----------
    paths = DisplayablePath.make_tree(
       Path('doc'),
       criteria=is_not_hidden
    )
    for path in paths:
       print(path.displayable())

    # With a criteria (skip hidden files)
    def is_not_hidden(path):
       return not path.name.startswith(".")

    paths = DisplayablePath.make_tree(Path('doc'))
    for path in paths:
       print(path.displayable())

    Example Output:
    ----------
    doc/
    ├── _static/
    │   ├── embedded/
    │   │   ├── deep_file
    │   │   └── very/
    │   │       └── deep/
    │   │           └── folder/
    │   │               └── very_deep_file
    │   └── less_deep_file
    ├── about.rst
    ├── conf.py
    └── index.rst
    """

    display_filename_prefix_middle = "├──"
    display_filename_prefix_last = "└──"
    display_parent_prefix_middle = "    "
    display_parent_prefix_last = "│   "

    def __init__(self, path: str, parent_path, is_last) -> None:

        self.path: str | Path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + "/"
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(
            list(path for path in root.iterdir() if criteria(path)),
            key=lambda s: str(s).lower(),
        )
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(
                    path, parent=displayable_root, is_last=is_last, criteria=criteria
                )
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + "/"
        return self.path.name

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (
            self.display_filename_prefix_last
            if self.is_last
            else self.display_filename_prefix_middle
        )

        parts = ["{!s} {!s}".format(_filename_prefix, self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(
                self.display_parent_prefix_middle
                if parent.is_last
                else self.display_parent_prefix_last
            )
            parent = parent.parent

        return "".join(reversed(parts))
