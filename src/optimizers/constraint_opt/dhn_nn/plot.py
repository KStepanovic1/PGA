from src.optimizers.constraint_opt.dhn_nn.common_py_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_param_imports import *
from src.optimizers.constraint_opt.dhn_nn.common_function_imports import *


class Plot:
    def __init__(self, result_p, plot_p):
        self.result_p = result_p
        self.plot_p = plot_p
        self.result_plnn_milp_p: Path = result_p.joinpath("plnn_milp")
        self.result_icnn_gd_p: Path = result_p.joinpath("icnn_gd")
        self.parent_p: Path = Path(__file__).parent
        with open(
            self.result_p.joinpath(
                "data_num_{}_heat_demand_real_world_for_L={}_time_interval_{}_max_Q_{}MW_deltaT_{}C.csv".format(
                    GridProperties["ConsumerNum"],
                    PipePreset1["Length"],
                    TimeParameters["TimeInterval"],
                    Generator1["MaxHeatProd"],
                    Generator1["MaxRampRateTemp"],
                )
            ),
            "rb",
        ) as f:
            self.data = pd.read_csv(f)[["Heat demand 1", "Electricity price"]]
        with open(
            self.result_p.joinpath("supply_pipe_plugs.pickle"),
            "rb",
        ) as f:
            self.supply_pipe_plugs = pickle.load(f)
        with open(
            self.result_p.joinpath("return_pipe_plugs.pickle"),
            "rb",
        ) as f:
            self.return_pipe_plugs = pickle.load(f)
        with open(
            Path(__file__).parent.joinpath("state_dict_" + "plnn" + ".pkl"), "rb"
        ) as f:
            self.state_dict = pickle.load(f)
        self.produced_heat_max = self.state_dict["Produced heat max"]
        self.produced_heat_min = self.state_dict["Produced heat min"]

    @staticmethod
    def plot_mark(ax, x, y, color, mark):
        """
        Plot mark on existing figure.
        """
        for i in range(len(x)):
            ax.plot(x[i], y[i], color, marker=mark, markersize=5)
        return ax

    @staticmethod
    def create_neuron_string(neuron) -> str:
        """
        Create string of the neuron depending on the type of neurons.
        For example, if the neuron is int (5) string is 5.
        If the neuron is list ([5,5]) string is 5_5.
        """
        neuron_ = ""
        if type(neuron) is int:
            neuron_ = str(neuron)
        elif type(neuron) is list:
            for nn in neuron:
                neuron_ += str(nn)
                neuron_ += "_"
            neuron_ = neuron_[:-1]
        else:
            print("Incorrect type of neuron!")
            exit(1)
        return neuron_

    @staticmethod
    def get_last_element(x):
        """
        Iterates through the list backwards, and gets the first element that is not nan.
        """
        y = []
        for i in range(len(x)):
            for j in range(1, len(x[i]) + 1):
                if not math.isnan(x[i][-j]):
                    y.append(x[i][-j])
                    break
        return y

    @staticmethod
    def summarize_validation_loss(
        val_loss_summary, val_loss_no_early_stop, val_loss_early_stop, fun_type
    ):
        """
        Calculate mean, upper and lower bounds of validation losses without and with early stopping.
        """
        val_loss = {}
        val_loss["early_stop"] = Plot.get_last_element(x=val_loss_early_stop)
        val_loss["no_early_stop"] = Plot.get_last_element(x=val_loss_no_early_stop)
        for regulator in ["early_stop", "no_early_stop"]:
            mean = sum(val_loss[regulator]) / len(val_loss[regulator])
            std_dev = np.std(np.array(val_loss[regulator]))
            val_loss_summary[fun_type][regulator]["mean"].append(mean)
            val_loss_summary[fun_type][regulator]["up_bound"].append(mean + std_dev)
            val_loss_summary[fun_type][regulator]["low_bound"].append(mean - std_dev)
        return val_loss_summary

    @staticmethod
    def sum(path, optimizer, coeff_day, column, null_solve=False):
        """
        Calculate sum of specific column for different days.
        """
        files = os.listdir(path)
        sum = 0
        for coeff in coeff_day:
            file = csv_file_finder(
                files=files,
                start=optimizer + "_" + str(coeff) + "_",
                null_solve=null_solve,
            )
            # no file found
            if file == None:
                sum = None
            else:
                # we start from the index time_delay[str(PipePreset1["Length"])] in order to mitigate effect of previous actions
                sum += np.sum(
                    np.array(pd.read_csv(path.joinpath(file))[column])[
                        time_delay[str(PipePreset1["Length"])] :
                    ]
                )
        return sum

    @staticmethod
    def average(path, optimizer, coeff_day, column, null_solve):
        """
        Calculate average of specific column for different days.
        """
        files = os.listdir(path)
        sum = 0
        for coeff in coeff_day:
            file = csv_file_finder(
                files=files,
                start=optimizer + "_" + str(coeff) + "_",
                null_solve=null_solve,
            )
            # no file found
            if file == None:
                sum = None
            else:
                sum += np.sum(
                    np.array(pd.read_csv(path.joinpath(file))[column])[
                        time_delay[str(PipePreset1["Length"])] :
                    ]
                )
                sum = sum / (
                    len(coeff_day)
                    * (
                        TimeParameters["PlanningHorizon"]
                        - time_delay[str(PipePreset1["Length"])]
                    )
                )
        return sum

    @staticmethod
    def average_actual_violation(path, optimizer, coeff_day, column, null_solve):
        """
        Calculate average of actual violation for the list of days specified in coeff_day.
        As all violation results are saved as percentage, we need to transform this percentages to violations in
        physics units (C, kg/s, MWh).
        """
        files = os.listdir(path)
        sum = 0
        for coeff in coeff_day:
            file = csv_file_finder(
                files=files, start=optimizer + "_" + str(coeff), null_solve=null_solve
            )
            sum_percent = np.sum(np.array(pd.read_csv(path.joinpath(file))[column]))
            if column == "Supply_inlet_violation":
                sum += sum_percent
            elif column == "Supply_outlet_violation":
                sum += sum_percent
            elif column == "Mass_flow_violation":
                sum += sum_percent
            elif column == "Delivered_heat_violation":
                sum += sum_percent
        return sum / (len(coeff_day) * TimeParameters["PlanningHorizon"])

    @staticmethod
    def standard_deviation(path, optimizer, coeff_day, column, null_solve):
        """
        Calculate standard deviation of specific column for all days.
        """
        files = os.listdir(path)
        temp = []
        for coeff in coeff_day:
            file = csv_file_finder(
                files=files, start=optimizer + "_" + str(coeff), null_solve=null_solve
            )
            temp.extend(list(pd.read_csv(path.joinpath(file))[column]))
        temp = np.array(temp)
        standard_deviation = np.std(temp)
        return standard_deviation

    @staticmethod
    def form_2D_dictionary(x, y):
        """
        Form dictionary inside the dictionary, where the first order key is the element of x, and the second order key is the element of y.
        The value is an empty list.
        """
        dictionary = {}
        for xx in x:
            dictionary[str(xx)] = {}
            for yy in range(y):
                dictionary[str(xx)][str(yy)] = []
        return dictionary

    @staticmethod
    def form_3D_dictionary(x, y, z):
        """
        Form dictionary inside the dictionary, where the first order key is the element of x, and the second order key is the element of y.
        The value is an empty list.
        """
        dictionary = {}
        for xx in x:
            dictionary[str(xx)] = {}
            for yy in y:
                dictionary[str(xx)][str(yy)] = {}
                for zz in z:
                    dictionary[str(xx)][str(yy)][str(zz)] = []
        return dictionary

    @staticmethod
    def form_3D_dictionary_mean_std(x, y, z):
        """
        Form dictionary inside the dictionary, where the first order key is the element of x, and the second order key is the element of y.
        The value is an empty list.
        """
        dictionary = {}
        for xx in x:
            dictionary[str(xx)] = {}
            for yy in y:
                dictionary[str(xx)][str(yy)] = {}
                for zz in z:
                    dictionary[str(xx)][str(yy)][str(zz)]["mean"] = []
                    dictionary[str(xx)][str(yy)][str(zz)]["std"] = []
        return dictionary

    @staticmethod
    def sum_cost_and_violations(
        operation_cost,
        supply_inlet_violation,
        supply_outlet_violation,
        mass_flow_violation,
        delivered_heat_violation,
        supply_inlet_violation_penalty,
        delivered_heat_penalty,
    ):
        """
        Sum operation costs and penalty*violations.
        """
        if (
            operation_cost is None
            and supply_inlet_violation is None
            and supply_outlet_violation is None
            and mass_flow_violation is None
            and delivered_heat_violation is None
        ):
            cost_plus_violation = None
        elif (
            operation_cost is not None
            and supply_inlet_violation is not None
            and supply_outlet_violation is not None
            and mass_flow_violation is not None
            and delivered_heat_violation is not None
        ):
            cost_plus_violation = (
                operation_cost
                + supply_inlet_violation_penalty * supply_inlet_violation
                + delivered_heat_penalty * supply_outlet_violation
                + delivered_heat_penalty * delivered_heat_violation
                + delivered_heat_penalty * mass_flow_violation
            )
        else:
            print(
                operation_cost,
                supply_inlet_violation,
                supply_outlet_violation,
                mass_flow_violation,
                delivered_heat_violation,
            )
            warnings.warn(
                "The result is None only if file has not been found. Having some files being None, and other not is an error!"
            )
            cost_plus_violation = np.inf
        return cost_plus_violation
