from ....command.test.vattenfall_ams.analysis_results import run_grid_different_profile


if __name__ == "__main__":
    total_time = 0
    n = 10
    for i in range(n):
        run_grid_different_profile()
