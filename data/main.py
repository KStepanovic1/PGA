from .data_processing import DataProcessing, DataRescaling, DataInterpolation


if __name__ == "__main__":
    max_heat_demand = 70  # [MW]
    interpolation_interval = 1800  # [s]
    data_scale = DataRescaling(max_heat_demand=max_heat_demand, min_heat_demand=0)
    data = data_scale.create_dataset()
    data_scale.save_data(
        data=data, name="processed_data_max_q_" + str(max_heat_demand) + ".csv"
    )
    """
    data_interpolation = DataInterpolation(
        interpolation_interval=interpolation_interval, max_heat_demand=max_heat_demand
    )
    data = data_interpolation.create_dataset()
    data_interpolation.save_data(
        data=data, name="processed_data_" + str(interpolation_interval) + "s.csv"
    )
    """
