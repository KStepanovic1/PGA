from .PL_utils import DataPaths, Data, NoUnstableIndices, Plots


if __name__ == "__main__":
    # data_1 = Data(182, 4000, [0, -1000], 25000000)
    # data_1._Data__generate_plot()
    data_2 = NoUnstableIndices(182, 4000, [0], 25000000)
    # data_2._Data__create_visual_csv()
    data_2._NoUnstableIndices__generate_plot()
    # data_2._NoUnstableIndices__plot_single_day(5)
