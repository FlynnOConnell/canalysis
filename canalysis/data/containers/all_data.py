from collections.abc import MutableMapping
from canalysis.helpers.wrappers import Singleton


@Singleton
class AllData(MutableMapping):
    """
    Custom mapping that works with properties from mutable object. Used to store each
    instance
    of data to iterate over and provide additional functionality.

    ..: Usage:
    alldata = AllData(function='xyz')
    and
    d.function returns 'xyz'
    """

    def __init__(self, *args, **kwargs):
        self.__dict__.update(*args, **kwargs)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __str__(self):
        return "\n".join(
            f"{key} - {len(value)} sessions." for key, value in self.__dict__.items()
        )

    def __repr__(self):
        return "\n".join(
            f"{key} - {len(value)} sessions." for key, value in self.__dict__.items()
        )
