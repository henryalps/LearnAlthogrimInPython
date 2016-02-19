# -*- coding: utf-8 -*-
# This class store all the methods related with plot operations.
import matplotlib.pyplot as plt
import numpy as np


class DrawToolkit:
    def __init__(self):
        plt.close("all")
        self.figNum = 0
        self.intervalNum = 4
        self.firstColor = 'blue'  # Main Color
        self.secondColor = 'green'  # Second Color
        self.assistColor = 'grey'  # Assistant Line color
        self.colorAbbr = ['r', 'g', 'b', 'c']

        self.calcPearsonR = True
        return

    def generate_scatter_plt(self, x, y, x_label, y_label, title):
        # init figure options
        fig, ax_scatter = plt.subplots(figsize=(8, 8))
        ax_scatter.set_xlabel(x_label)
        ax_scatter.set_ylabel(y_label)
        ax_scatter.set_title(title)
        counts_in_each_error_interval = [0] * self.intervalNum

        # calculate range first
        x_y_min = min(min(x), min(y))
        x_y_max = max(max(x), max(y))
        x_y_shift = (x_y_max - x_y_min) / 20
        x_y_min -= x_y_shift
        x_y_max += x_y_shift

        # set range and plot scatter
        ax_scatter.set_xlim((x_y_min, x_y_max))
        ax_scatter.set_ylim((x_y_min, x_y_max))
        area = np.pi * 4 ** 2  # radius is 4
        for point_x, point_y in zip(x, y):
            shift = int(np.float(abs(point_y - point_x) / 5.0))
            shift = self.intervalNum - 1 if(shift >= self.intervalNum ) else shift
            for index in range(shift, counts_in_each_error_interval.__len__()):
                counts_in_each_error_interval[index] += 1
            ax_scatter.scatter(point_x, point_y, s=area,
                               c=self.colorAbbr[shift], alpha=0.5)

        # calculate the percent of points in each interval
        counts_in_each_error_interval[:] = [100 * element / len(x) for element in counts_in_each_error_interval]

        # plot y=x line
        ax_scatter.plot([x_y_min, x_y_max], [x_y_min, x_y_max],
                        self.colorAbbr[0] + '--', label='Perfect Regression')  # , label='Random guess'
        # plot y=x + 5 and y = x - 5 line
        ax_scatter.plot([x_y_min, x_y_max], [x_y_min + 5, x_y_max + 5],
                        self.colorAbbr[1] + '--', label=('Error <= 5 mmHg:' + '%.2f'
                                                         % (counts_in_each_error_interval[0]) + '%'))
        ax_scatter.plot([x_y_min, x_y_max], [x_y_min - 5, x_y_max - 5],
                        self.colorAbbr[1] + '--')  # , label='Random guess'
        # plot y=x + 10 and y = x - 10 line
        ax_scatter.plot([x_y_min, x_y_max], [x_y_min + 10, x_y_max + 10],
                        self.colorAbbr[2] + '--', label=('Error <= 10 mmHg:' + '%.2f' %
                                                         (counts_in_each_error_interval[1]) + ' %'))
        ax_scatter.plot([x_y_min, x_y_max], [x_y_min - 10, x_y_max - 10],
                        self.colorAbbr[2] + '--')  # , label='Random guess'
        # plot y=x + 15 and y = x - 15 line
        ax_scatter.plot([x_y_min, x_y_max], [x_y_min + 15, x_y_max + 15],
                        self.colorAbbr[3] + '--', label=('Error <= 15 mmHg:' + '%.2f' %
                                                         (counts_in_each_error_interval[2]) + ' %'))
        ax_scatter.plot([x_y_min, x_y_max], [x_y_min - 15, x_y_max - 15],
                        self.colorAbbr[3] + '--')  # , label='Random guess'

        ax_scatter.legend()

        plt.tight_layout()
        return plt
