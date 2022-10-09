from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class CalFigure(Figure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tight_layout()
        self.set_dpi(300)

    def close(self):
        plt.close(self)

    def save(self, dir):
        plt.savefig(dir)







