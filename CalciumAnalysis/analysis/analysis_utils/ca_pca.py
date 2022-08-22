from __future__ import annotations

from typing import Optional, Iterable, Any, ClassVar, Sized
import numpy as np
import pandas as pd
from utils import excepts as e
from graphs.plot import ScatterPlots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class CaPrincipalComponentsAnalysis:
    def __init__(
            self,
            data: pd.DataFrame,
            colors: Sized[Any],
            numcomp: Optional[int] = 4
    ):
        self.data: pd.DataFrame = data
        self.colors: Sized[Any] = colors
        self.numcomp: int = numcomp
        self.pca: ClassVar = None
        self.variance_explained: Iterable[Any] | None = None
        self.pca_df: pd.DataFrame | None = None
        self.fit_pca()

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

    def get_plots(self, color_dict: dict, **kwargs) -> ClassVar[ScatterPlots]:
        return ScatterPlots(self.pca_df, self.colors, color_dict, **kwargs)

