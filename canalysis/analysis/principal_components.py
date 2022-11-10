from __future__ import annotations

from typing import Optional, Iterable, Any, ClassVar, Sized
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_pca(data, numcomp: int = 4):
    return _PrincipalComponents(data, numcomp=3)


class _PrincipalComponents:
    def __init__(
            self,
            data: pd.DataFrame,
            numcomp: int
    ):
        self.data: pd.DataFrame = data
        self.numcomp: int = numcomp
        self.pca: ClassVar = None
        self.variance_explained: Iterable[Any] | None = None
        self.pca_df: pd.DataFrame | None = None
        self.fit_pca()

    def __repr__(self):
        return f"{type(self).__name__}, {self.numcomp}"

    def fit_pca(self, ) -> None:
        data_prepped = StandardScaler().fit_transform(self.data)
        self.pca = PCA(n_components=self.numcomp)
        data_fit = self.pca.fit_transform(data_prepped)
        self.variance_explained = np.round(self.pca.explained_variance_ratio_ * 100, decimals=1)
        labels = [
            "PC" + str(x) + f" - {self.variance_explained[x - 1]}%"
            for x in range(1, len(self.variance_explained) + 1)
        ]
        self.pca_df = pd.DataFrame(data_fit, columns=labels)
        return None

