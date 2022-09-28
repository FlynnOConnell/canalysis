import matplotlib


class CalFigure(matplotlib.figure.Figure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tight_layout()
        self.set_dpi(300)


