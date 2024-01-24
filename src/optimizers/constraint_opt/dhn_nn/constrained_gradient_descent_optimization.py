import copy
import math
import tensorflow as tf
import time
import statistics
import numpy as np
import pandas as pd
import scipy

from pathlib import Path
from matplotlib import pyplot as plt


def calculate_euclidian_distance(var1, var2):
    """
    Calculate euclidian distance between two variables.
    """
    d = 0
    for i in range(len(var1)):
        d += (var1[i] - var2[i]) ** 2
    d = math.sqrt(d)
    return d


class FunctionOne:
    """
    Function one with exponentials.
    """

    def __init__(self, a_1, a_2, a_3, C, q):
        self.num_con = 3
        self.C = C
        self.a_1 = a_1
        self.a_2 = a_2
        self.a_3 = a_3
        self.q = q
        self.optimal_solution = [3.4774, 2.15554, 0.877075]
        self.J = (
            self.optimal_solution[0]
            + self.optimal_solution[1]
            + self.optimal_solution[2]
        )
        self.result_p: Path = (
            Path(__file__).parents[4]
            / "results/constraint_opt/const_gradient_descent_optimization"
        )
        self.plot_p: Path = (
            Path(__file__).parents[4]
            / "plots/constraint_opt/const_gradient_descent_optimization/Toy problem"
        )

    @tf.function
    def calculate_gradient(self, var, epsilon):
        """
        Calculate gradient over specified function.
        """
        with tf.GradientTape() as tape:
            tape.watch(var)
            f = (
                self.a_1 * var[:, 0]
                + self.a_2 * var[:, 1]
                + self.a_3 * var[:, 2]
                + self.C
                * (tf.math.exp(0.1 + 0.75 * var[:, 0]) - self.q[0] - epsilon[0]) ** 2
                + self.C
                * (
                    tf.math.exp(0.05 + var[:, 0] + 0.5 * var[:, 1])
                    - self.q[1]
                    - epsilon[1]
                )
                ** 2
                + self.C
                * (
                    tf.math.exp(0.1 * var[:, 0] + 0.5 * var[:, 1] + var[:, 2])
                    - self.q[2]
                    - epsilon[2]
                )
                ** 2
            )
        grads = tape.gradient(f, var)
        return grads

    @tf.function
    def calculate_gradient_ipdd(self, var, lam, C):
        """
        Calculate gradient over specified function.
        """
        with tf.GradientTape() as tape:
            tape.watch(var)
            f = (
                self.a_1 * var[:, 0]
                + self.a_2 * var[:, 1]
                + self.a_3 * var[:, 2]
                + lam[0] * (tf.math.exp(0.1 + 0.75 * var[:, 0]) - self.q[0])
                + lam[1] * (tf.math.exp(0.05 + var[:, 0] + 0.5 * var[:, 1]) - self.q[1])
                + lam[2]
                * (
                    tf.math.exp(0.1 * var[:, 0] + 0.5 * var[:, 1] + var[:, 2])
                    - self.q[2]
                )
                + C * (tf.math.exp(0.1 + 0.75 * var[:, 0]) - self.q[0]) ** 2
                + C * (tf.math.exp(0.05 + var[:, 0] + 0.5 * var[:, 1]) - self.q[1]) ** 2
                + C
                * (
                    tf.math.exp(0.1 * var[:, 0] + 0.5 * var[:, 1] + var[:, 2])
                    - self.q[2]
                )
                ** 2
            )
        grads = tape.gradient(f, var)
        return grads

    def calculate_constraint_violations_list(self, var):
        """
        Calculate value of the maximum constraint violation.
        var: list
        """
        equation_one = math.exp(0.1 + 0.75 * var[0]) - self.q[0]
        equation_two = math.exp(0.05 + var[0] + 0.5 * var[1]) - self.q[1]
        equation_three = math.exp(0.1 * var[0] + 0.5 * var[1] + var[2]) - self.q[2]
        return [equation_one, equation_two, equation_three]

    def calculate_objective_function(self, var):
        """
        Value of objective function.
        var:list
        """
        J = self.a_1 * var[0] + self.a_2 * var[1] + self.a_3 * var[2]
        return J

    def calculate_approximate_objective_function(self, var, constraint_violations):
        """
        Value of approximate objective function.
        var:list, constraint_violations:list
        """
        hat_J = (
            self.a_1 * var[0]
            + self.a_2 * var[1]
            + self.a_3 * var[2]
            + self.C * (constraint_violations[0]) ** 2
            + self.C * (constraint_violations[1]) ** 2
            + self.C * (constraint_violations[2]) ** 2
        )
        return hat_J

    def determine_continue_criteria(
        self,
        constraint_violation,
        constraint_violations,
        stop_threshold,
        const_threshold,
        iter_const_threshold,
        convergence_iter,
    ):
        """
        Stop iterating when all constraint violations are less than threshold (stop threshold)
        or when certain constraint violations do not change for more than threshold (const threshold)
        for more than iter_const_threshold iterations.
        """
        stop_criteria = []
        for i in range(self.num_con):
            k = 0
            if abs(constraint_violation[i]) <= stop_threshold:
                stop_criteria.append(True)
            else:
                if convergence_iter <= iter_const_threshold:
                    stop_criteria.append(False)
                    break
                else:
                    for j in range(1, iter_const_threshold):
                        if (
                            abs(
                                constraint_violations["constraint {}".format(i)][
                                    convergence_iter - j - 1
                                ]
                                - constraint_violations["constraint {}".format(i)][
                                    max(convergence_iter - j - 1 - 1, 0)
                                ]
                            )
                            <= const_threshold
                        ):
                            k += 1
                        else:
                            break
                    if k == iter_const_threshold - 1:
                        stop_criteria.append(True)
                    else:
                        stop_criteria.append(False)
                        break
        if all(stop_criteria):
            continue_criteria = False
        else:
            continue_criteria = True
        return continue_criteria

    def optimize_with_epsilon(self, initial_vector, grad_iter_max, delta, patience):
        """
        Calculate variable through adding epsilon.

        It would be great if we could use the warm start up -- at iteration k+1, start the optimization
        from the vector optimization converged to at iteration k.
        However, starting from this vector takes more gradient descent steps than starting from initial vector.
        For this to be fixed, instead of resetting Adam optimizer, we should save its weights at the end of k iteration and re-initialize
        optimizer at iteration k+1 with them.
        However, that is impossible: using save_own_variables and later loading those variables does not work because optimizer start with only one variable
        and adds new variables when it starts the optimization. Even if you already add those variables manually, it will not re-initialize them,
        but instead add new variables.
        Function build of optimizer does not work. Source code has TO DO note https://github.com/keras-team/keras/blob/v2.14.0/keras/optimizers/optimizer.py#L402-L426
        """
        convergence_iter = 0
        continue_criteria = True
        epsilon_iter = [0, 0, 0]
        epsilon, J, hat_J, grad_iter_, execution_time = [], [], [], [], []
        execution_time_sum = 0
        constraint_violations = {}
        for i in range(self.num_con):
            constraint_violations["constraint {}".format(i)] = []
        while execution_time_sum<100:
            convergence_iter += 1
            grad_iter, grad_patience_iter = 0, 0
            var = tf.Variable(np.array(initial_vector).reshape(1, -1), dtype=tf.float32)
            opt = tf.keras.optimizers.Adam(learning_rate=0.01)
            grads_ = []
            grads_prev = tf.convert_to_tensor(
                np.array([100, 100, 100], dtype=np.float32).reshape(1, self.num_con)
            )
            start_time = time.time()
            while grad_patience_iter < patience and grad_iter < grad_iter_max:
                grads = self.calculate_gradient(var, epsilon=epsilon_iter)
                delta_tf = tf.abs(grads_prev - grads)
                if tf.reduce_any(delta_tf < delta, 1):
                    grad_patience_iter += 1
                else:
                    grad_patience_iter = 0
                grads_prev = grads
                grads_.append(list(np.array(grads)[0]))
                zipped = zip([grads], [var])
                opt.apply_gradients(zipped)
                grad_iter += 1
            # FunctionOne.plot_grads(grads = grads_)
            var = list(np.array(var[0]))
            # initial_vector = var
            # calculate all constraint violations
            constraint_violation = self.calculate_constraint_violations_list(var)
            # calculate continue criteria
            continue_criteria = self.determine_continue_criteria(
                constraint_violation,
                constraint_violations,
                stop_threshold=0.01,
                const_threshold=0.01,
                iter_const_threshold=5,
                convergence_iter=convergence_iter,
            )
            for i in range(self.num_con):
                epsilon_iter[i] = epsilon_iter[i]- (2 / convergence_iter) * constraint_violation[i]
                constraint_violations["constraint {}".format(i)].append(
                    constraint_violation[i]
                )
            end_time = time.time()
            execution_time.append(end_time - start_time)
            execution_time_sum+=end_time - start_time
            J_iter = self.calculate_objective_function(var)
            J.append(J_iter)
            hat_J_iter = self.calculate_approximate_objective_function(
                var, constraint_violation
            )
            hat_J.append(hat_J_iter)
            grad_iter_.append(grad_iter)
            print("Variable is:", var)
            print("Constraint violations are:", constraint_violation)
            print("J={}".format(J_iter))
            print("hat_J={}".format(hat_J_iter))
        self.save_results(
            var=var,
            epsilon=epsilon,
            n_iter=convergence_iter,
            J=J,
            hat_J=hat_J,
            constraint_violations=constraint_violations,
            execution_time=execution_time,
            grad_iter=grad_iter_,
            extension="gdco_no_neg",
        )

    def ipdd_algorithm(
        self, initial_vector, lam, C, eta, grad_iter_max, delta, patience
    ):
        """
        Calculate variable through IPDD algorithm.
        """
        convergence_iter, execution_time_sum = 0, 0
        continue_criteria = True
        J, hat_J, grad_iter_, execution_time = [], [], [], []
        constraint_violations = {}
        for i in range(self.num_con):
            constraint_violations["constraint {}".format(i)] = []
        while execution_time_sum<100:
            convergence_iter += 1
            grad_iter, grad_patience_iter = 0, 0
            var = tf.Variable(np.array(initial_vector).reshape(1, -1), dtype=tf.float32)
            opt = tf.keras.optimizers.Adam(learning_rate=0.01)
            grads_ = []
            grads_prev = tf.convert_to_tensor(
                np.array([100, 100, 100], dtype=np.float32).reshape(1, self.num_con)
            )
            start_time = time.time()
            while grad_patience_iter < patience and grad_iter < grad_iter_max:
                grads = self.calculate_gradient_ipdd(var, C=C, lam=lam)
                delta_tf = tf.abs(grads_prev - grads)
                if tf.reduce_any(delta_tf < delta, 1):
                    grad_patience_iter += 1
                else:
                    grad_patience_iter = 0
                grads_prev = grads
                grads_.append(list(np.array(grads)[0]))
                zipped = zip([grads], [var])
                opt.apply_gradients(zipped)
                grad_iter += 1
            # FunctionOne.plot_grads(grads = grads_)
            var = list(np.array(var[0]))
            # initial_vector = var
            # calculate all constraint violations
            constraint_violation = self.calculate_constraint_violations_list(var)
            # calculate continue criteria
            continue_criteria = self.determine_continue_criteria(
                constraint_violation,
                constraint_violations,
                stop_threshold=0.01,
                const_threshold=0.01,
                iter_const_threshold=5,
                convergence_iter=convergence_iter,
            )
            for i in range(self.num_con):
                lam[i] = lam[i] + C * constraint_violation[i]
                constraint_violations["constraint {}".format(i)].append(
                    constraint_violation[i]
                )
            C = eta * C
            end_time = time.time()
            execution_time.append(end_time - start_time)
            execution_time_sum+= end_time - start_time
            J_iter = self.calculate_objective_function(var)
            J.append(J_iter)
            hat_J_iter = self.calculate_approximate_objective_function(
                var, constraint_violation
            )
            hat_J.append(hat_J_iter)
            grad_iter_.append(grad_iter)
            print("Variable is:", var)
            print("Constraint violations are:", constraint_violation)
            print("J={}".format(J_iter))
            print("hat_J={}".format(hat_J_iter))
        self.save_results(
            var=var,
            epsilon=[0],
            n_iter=convergence_iter,
            J=J,
            hat_J=hat_J,
            constraint_violations=constraint_violations,
            execution_time=execution_time,
            grad_iter=grad_iter_,
            extension="ipdd",
        )

    def cal_initialization_impact(
        self, initial_vectors, grad_iter_max, delta, patience
    ):
        """
        For each initial vector, plot maximal Euclidian distance of corresponding solution
        from all other solutions.
        """
        N = len(initial_vectors)
        solutions = []
        max_euclidian_distance = []
        for initial_vector in initial_vectors:
            var = tf.Variable(np.array(initial_vector).reshape(1, -1), dtype=tf.float32)
            opt = tf.keras.optimizers.Adam(learning_rate=0.01)
            grad_iter, grad_patience_iter = 0, 0
            grads_prev = tf.convert_to_tensor(
                np.array([100, 100, 100], dtype=np.float32).reshape(1, self.num_con)
            )
            while grad_patience_iter < patience and grad_iter < grad_iter_max:
                grads = self.calculate_gradient(
                    var, epsilon=[0] * self.num_con, epsilon_coeff=1
                )
                delta_tf = tf.abs(grads_prev - grads)
                if tf.reduce_any(delta_tf < delta, 1):
                    grad_patience_iter += 1
                else:
                    grad_patience_iter = 0
                grads_prev = grads
                zipped = zip([grads], [var])
                opt.apply_gradients(zipped)
                grad_iter += 1
            var = list(np.array(var[0]))
            solutions.append(var)
        for i in range(N):
            t_max = 0
            for j in range(N):
                t = calculate_euclidian_distance(solutions[i], solutions[j])
                if t > t_max:
                    t_max = t
            # maximum euclidian distance for each solution from other solutions
            max_euclidian_distance.append(t_max)
        plt.figure(figsize=(15, 5))
        plt.bar(
            range(N),
            max_euclidian_distance,
            tick_label=[str(initial_vector) for initial_vector in initial_vectors],
        )
        plt.xlabel("Initialization points")
        plt.ylabel("Euclidian distance")
        plt.show()

    def save_results(
        self,
        var,
        epsilon,
        n_iter,
        J,
        hat_J,
        constraint_violations,
        execution_time,
        grad_iter,
        extension,
    ):
        """
        Save results
        var: list
        epsilon: list
        J: list
        hat_J: list
        constraint_violations: dict
        """
        var_df = pd.DataFrame(var)
        n_iter_df = pd.DataFrame([n_iter])
        epsilon_df = pd.DataFrame(epsilon)
        J_df = pd.DataFrame(J)
        hat_J_df = pd.DataFrame(hat_J)
        constraint_violations_df = pd.DataFrame(constraint_violations)
        execution_time_df = pd.DataFrame(execution_time)
        grad_iter_df = pd.DataFrame(grad_iter)
        var_df.to_csv(self.result_p.joinpath("Solution_C_{}_".format(self.C) + extension + ".csv"))
        epsilon_df.to_csv(self.result_p.joinpath("Epsilons_C_{}_".format(self.C) + extension + ".csv"))
        n_iter_df.to_csv(
            self.result_p.joinpath("Number_of_iterations_C_{}_".format(self.C) + extension + ".csv")
        )
        J_df.to_csv(self.result_p.joinpath("J_C_{}_".format(self.C) + extension + ".csv"))
        hat_J_df.to_csv(self.result_p.joinpath("hat_J_C_{}_".format(self.C) + extension + ".csv"))
        constraint_violations_df.to_csv(
            self.result_p.joinpath("Constraint_violations_C_{}_".format(self.C) + extension + ".csv")
        )
        execution_time_df.to_csv(
            self.result_p.joinpath("Execution_time_C_{}_".format(self.C) + extension + ".csv")
        )
        grad_iter_df.to_csv(
            self.result_p.joinpath(
                "Number_of_gradient_descent_iterations_C_{}_".format(self.C) + extension + ".csv"
            )
        )

    @staticmethod
    def plot_grads(grads):
        """
        Plot gradients.
        """
        grads_x = [sublist[0] for sublist in grads]
        grads_y = [sublist[1] for sublist in grads]
        grads_z = [sublist[2] for sublist in grads]

        plt.plot(grads_x)
        plt.title("Gradient x")
        plt.show()

        plt.plot(grads_x[15000:])
        plt.title("Gradient x zoom")
        plt.show()

        plt.plot(grads_y)
        plt.title("Gradient y")
        plt.show()

        plt.plot(grads_y[15000:])
        plt.title("Gradient y zoom")
        plt.show()

        plt.plot(grads_z)
        plt.title("Gradient z")
        plt.show()

        plt.plot(grads_z[15000:])
        plt.title("Gradient z zoom")
        plt.show()

    def plot_computational_time(self):
        """
        Plot objective function, approximate objective function,
        constraint values as the function of computational time.
        """
        result_p: Path = (
            Path(__file__).parents[4]
            / "results/constraint_opt/const_gradient_descent_optimization"
        )
        J = {"gdco": [], "ipdd": []}
        hat_J = {"gdco": [], "ipdd": []}
        constraint_violations = {"gdco": {}, "ipdd": {}}
        for i in range(self.num_con):
            constraint_violations["gdco"]["constraint {}".format(i)] = []
            constraint_violations["ipdd"]["constraint {}".format(i)] = []
        constraint_violations_max = {"gdco": [], "ipdd": []}
        computational_time = {"gdco": {}, "ipdd": {}}
        J["gdco"] = list(pd.read_csv(result_p.joinpath("J_gdco.csv"))["0"])
        J["ipdd"] = list(pd.read_csv(result_p.joinpath("J_ipdd.csv"))["0"])
        hat_J["gdco"] = list(pd.read_csv(result_p.joinpath("hat_J_gdco.csv"))["0"])
        hat_J["ipdd"] = list(pd.read_csv(result_p.joinpath("hat_J_ipdd.csv"))["0"])
        for i in range(self.num_con):
            constraint_violations["gdco"]["constraint {}".format(i)] = list(
                pd.read_csv(result_p.joinpath("Constraint_violations_gdco.csv"))[
                    "constraint {}".format(i)
                ]
            )
            constraint_violations["ipdd"]["constraint {}".format(i)] = list(
                pd.read_csv(result_p.joinpath("Constraint_violations_ipdd.csv"))[
                    "constraint {}".format(i)
                ]
            )
        x = list(pd.read_csv(result_p.joinpath("Execution_time_gdco.csv"))["0"])
        computational_time["gdco"] = [sum(x[: i + 1]) for i in range(len(x))]
        y = list(pd.read_csv(result_p.joinpath("Execution_time_ipdd.csv"))["0"])
        computational_time["ipdd"] = [sum(y[: i + 1]) for i in range(len(y))]
        for i in range(len(J["gdco"])):
            # Extract the ith element from each list and find the absolute max
            max_value = float("-inf")
            max_constraint_index = None
            for constraint_number in constraint_violations["gdco"]:
                if abs(constraint_violations["gdco"][constraint_number][i]) > max_value:
                    max_value = abs(constraint_violations["gdco"][constraint_number][i])
                    max_constraint_index = constraint_number
            constraint_violations_max["gdco"].append(
                constraint_violations["gdco"][max_constraint_index][i]
            )
        for i in range(len(J["ipdd"])):
            # Extract the ith element from each list and find the absolute max
            max_value = float("-inf")
            max_constraint_index = None
            for constraint_number in constraint_violations["ipdd"]:
                if abs(constraint_violations["ipdd"][constraint_number][i]) > max_value:
                    max_value = abs(constraint_violations["ipdd"][constraint_number][i])
                    max_constraint_index = constraint_number
            constraint_violations_max["ipdd"].append(
                constraint_violations["ipdd"][max_constraint_index][i]
            )
        """
        if computational_time["gdco"][-1] < computational_time["ipdd"][-1]:
            for index, value in enumerate(computational_time["ipdd"]):
                if value > computational_time["gdco"][-1]:
                    computational_time["ipdd"] = computational_time["ipdd"][
                        : (index + 1)
                    ]
                    J["ipdd"] = J["ipdd"][: (index + 1)]
                    hat_J["ipdd"] = hat_J["ipdd"][: (index + 1)]
                    constraint_violations_max["ipdd"] = constraint_violations_max[
                        "ipdd"
                    ][: (index + 1)]
        """
        # objective function
        # plt.figure(figsize=(20, 5))
        plt.plot(computational_time["gdco"][0], J["gdco"][0], color="#B6C800", marker="P", markersize=15,
                 linestyle="dashed", label="PM")
        plt.axhline(y=J["gdco"][0], xmin=computational_time["gdco"][0]/100, xmax=100/100, color="#B6C800", linestyle="dashed")
        plt.plot(5, self.J, color="#16502E", marker="v", markersize=10, linestyle="dotted", label="NLP")
        plt.axhline(y=self.J, xmin=5/100, xmax=100/100, color="#16502E", linestyle="dotted")
        plt.plot(
            computational_time["gdco"],
            J["gdco"],
            color="#3374FF",
            marker="o",
            markersize=10,
            label="PGA",
            linestyle = "solid"
        )
        plt.plot(
            computational_time["ipdd"],
            J["ipdd"],
            color="#E31D1D",
            marker="*",
            markersize=10,
            label="IPDD",
            linestyle="dashdot"
        )
        plt.xlim([0, 100])
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Computational time [s]", fontsize=14)
        plt.ylabel("Objective value", fontsize=14)
        plt.legend()
        #plt.tight_layout()
        plt.savefig(
               self.plot_p.joinpath(
                    "Objective_value.pdf"
                )
        )
        plt.show()
        # approximate objective function
        # plt.figure(figsize=(20, 5))
        plt.plot(
            computational_time["gdco"],
            hat_J["gdco"],
            color="#3374FF",
            marker="o",
            markersize=10,
            label="PGDA",
            linestyle="solid"
        )
        plt.plot(
            computational_time["ipdd"],
            hat_J["ipdd"],
            color="#E31D1D",
            marker="*",
            markersize=10,
            label="IPDD",
            linestyle="dashdot"
        )
        plt.axhline(y=self.J, color="#33754E", linestyle="--", label=r"NLP$_{T_{lim}=100s}$")
        plt.xlabel("Computational time [s]", fontsize=11)
        plt.ylabel("Approximate Objective Value", fontsize=11)
        plt.xlim([0, 100])
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend()
        plt.savefig(
                self.plot_p.joinpath(
                    "Approximate_objective_value.pdf"
                )
        )
        plt.show()
        # constraint violations
        # plt.figure(figsize=(20, 5))
        plt.plot(computational_time["gdco"][0], constraint_violations_max["gdco"][0], color="#B6C800", marker="P",
                 markersize=15, linestyle="dashed", label="PM")
        plt.axhline(y=constraint_violations_max["gdco"][0], xmin=computational_time["gdco"][0] / 100, xmax=100 / 100, color="#B6C800",
                    linestyle="dashed")
        plt.plot(5, 0, color="#16502E", marker="v", markersize=10, linestyle="dotted", label="NLP")
        plt.axhline(y=0, xmin=5/100, xmax=1, color="#16502E", linestyle="dotted")
        plt.plot(
            computational_time["gdco"],
            constraint_violations_max["gdco"],
            color="#3374FF",
            marker="o",
            markersize=10,
            label="PGA",
        )
        plt.plot(
            computational_time["ipdd"],
            constraint_violations_max["ipdd"],
            color="#E31D1D",
            marker="*",
            markersize=10,
            label="IPDD",
            linestyle="dashdot"
        )
        plt.xlabel("Computational time [s]", fontsize=14)
        plt.ylabel("Constraint value", fontsize=14)
        plt.xlim([0, 100])
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend()
        plt.savefig(
                self.plot_p.joinpath(
                    "Constraint_value.pdf"
                )
        )
        plt.show()

    def mean_std_computational_time(self):
        """
        Print mean and standard deviation of computational time per outer iteration.
        """
        computation_time_gdco = list(
            pd.read_csv(self.result_p.joinpath("Execution_time_gdco.csv"))["0"]
        )
        computation_time_ipdd = list(
            pd.read_csv(
                self.result_p.joinpath("Execution_time_ipdd.csv")
            )["0"]
        )
        N = min(len(computation_time_gdco), len(computation_time_ipdd))
        print(scipy.stats.pearsonr(computation_time_gdco[:N], computation_time_ipdd[:N]))
        mean_gdco = statistics.mean(computation_time_gdco)
        std_gdco = statistics.stdev(computation_time_gdco)
        mean_ipdd = statistics.mean(computation_time_ipdd)
        std_ipdd = statistics.stdev(computation_time_ipdd)
        print("GDCO mean is {} and std is {}".format(mean_gdco, std_gdco))
        print("IPDD mean is {} and std is {}".format(mean_ipdd, std_ipdd))

    def plot_gradient_descent_initialization_vector(self, epsilon_coeffs):
        """
        Plot influence of gradient descent initial vector on the optimization.
        """
        for epsilon_coeff in epsilon_coeffs:
            num_iter_fix_initial = list(
                pd.read_csv(
                    self.result_p.joinpath(
                        "Number_of_gradient_descent_iterations_epsilon_coeff={}.csv".format(
                            epsilon_coeff
                        )
                    )
                )["0"]
            )
            num_iter_var_initial = list(
                pd.read_csv(
                    self.result_p.joinpath(
                        "Number_of_gradient_descent_iterations_var_initial_epsilon_coeff={}.csv".format(
                            epsilon_coeff
                        )
                    )
                )["0"]
            )
            N = max(len(num_iter_fix_initial), len(num_iter_var_initial))
            # number of gradient descent updates per iteration
            plt.plot(
                range(N),
                num_iter_fix_initial + [None] * (N - len(num_iter_fix_initial)),
                color="r",
                marker="o",
                label=r"$u^0 = c$",
            )
            plt.plot(
                range(N),
                num_iter_var_initial + [None] * (N - len(num_iter_var_initial)),
                color="b",
                marker="*",
                label=r"$u^0 = u^{i-1}$",
            )
            plt.xlabel("Iterations")
            plt.ylabel("Gradient descent iteration")
            plt.legend()
            plt.savefig(
                self.plot_p.joinpath(
                    "Number of gradient descent iterations per update iteration for epsilon_coeff={}.png".format(
                        epsilon_coeff
                    )
                )
            )
            plt.show()
            J_c, J_i_1, constraint_values_c, constraint_values_i_1 = {}, {}, {}, {}
        for epsilon_coeff in epsilon_coeffs:
            J_c["epsilon_coeff={}".format(epsilon_coeff)] = 0
            J_i_1["epsilon_coeff={}".format(epsilon_coeff)] = 0
            constraint_values_c["epsilon_coeff={}".format(epsilon_coeff)] = {}
            constraint_values_i_1["epsilon_coeff={}".format(epsilon_coeff)] = {}
        for epsilon_coeff in epsilon_coeffs:
            for i in range(self.num_con):
                constraint_values_c["epsilon_coeff={}".format(epsilon_coeff)] = []
                constraint_values_i_1["epsilon_coeff={}".format(epsilon_coeff)] = []
        for epsilon_coeff in epsilon_coeffs:
            J_c["epsilon_coeff={}".format(epsilon_coeff)] = list(
                pd.read_csv(
                    self.result_p.joinpath(
                        "J_epsilon_coeff={}.csv".format(epsilon_coeff)
                    )
                )["0"]
            )[-1]
            J_i_1["epsilon_coeff={}".format(epsilon_coeff)] = list(
                pd.read_csv(
                    self.result_p.joinpath(
                        "J_var_initial_epsilon_coeff={}.csv".format(epsilon_coeff)
                    )
                )["0"]
            )[-1]
            for i in range(self.num_con):
                constraint_values_c["epsilon_coeff={}".format(epsilon_coeff)].append(
                    list(
                        pd.read_csv(
                            self.result_p.joinpath(
                                "Constraint_violations_epsilon_coeff={}.csv".format(
                                    epsilon_coeff
                                )
                            )
                        )["constraint {}".format(i)]
                    )[-1]
                )
            constraint_values_i_1["epsilon_coeff={}".format(epsilon_coeff)].append(
                list(
                    pd.read_csv(
                        self.result_p.joinpath(
                            "Constraint_violations_var_initial_epsilon_coeff={}.csv".format(
                                epsilon_coeff
                            )
                        )
                    )["constraint {}".format(i)]
                )[-1]
            )
        bar_width = 0.20
        x_values = range(len(epsilon_coeffs))
        y_values = [x + bar_width for x in x_values]
        # objective function
        plt.figure(figsize=(20, 5))
        plt.bar(x_values, J_c.values(), color="r", width=bar_width, label=r"$u^0 = c$")
        plt.bar(
            y_values,
            J_i_1.values(),
            color="b",
            width=bar_width,
            label=r"$u^0 = u^{i-1}$",
        )
        plt.axhline(y=self.J, color="g", linestyle="--", label=r"$J^*$")
        # Set x-axis ticks and labels for each bar
        combined_labels = [
            r"$\eta={}$".format(epsilon_coeffs[i]) for i in range(len(epsilon_coeffs))
        ]
        plt.xticks([x + bar_width / 2 for x in x_values], combined_labels)
        plt.ylabel("Objective function")
        plt.legend()
        # plt.savefig(self.plot_p.joinpath("Objective_function.png"))
        plt.show()


if __name__ == "__main__":
    a_1 = 1
    a_2 = 1
    a_3 = 1
    C = 0.5
    grad_iter_max = 25000
    fun_one = FunctionOne(a_1=a_1, a_2=a_2, a_3=a_3, C=C, q=[15, 100, 10])
    # fun_one.optimize_linear_interpolation()
    initial_vectors = [
        [4, 2, 2],
        [4, 3, 2],
        [4, 4, 4],
        [4, 5, 4],
        [5, 3, 4],
        [5, 4, 4],
        [5, 5, 5],
    ]
    """
    fun_one.cal_initialization_impact(
        initial_vectors=initial_vectors,
        grad_iter_max=grad_iter_max,
        delta=0.000001,
        patience=50,
    )
    """
    fun_one.optimize_with_epsilon(
        initial_vector=initial_vectors[0],
        grad_iter_max=grad_iter_max,
        delta=0.000001,
        patience=50,
    )
    """
    fun_one.ipdd_algorithm(
        initial_vector=initial_vectors[0],
        lam=[0, 0, 0],
        C=C,
        eta=1.2,
        grad_iter_max=grad_iter_max,
        delta=0.000001,
        patience=50,
    )
    fun_one.plot_computational_time()
    """
