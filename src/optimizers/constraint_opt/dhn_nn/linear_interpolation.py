def calculate_epsilon_zero(var):
    """
    Maximum overshooting of the constraint.
    """
    equation_one, equation_two, equation_three = FunctionOne.calculate_equation_value(
        var=var
    )
    epsilon = float(
        tf.maximum(tf.maximum(equation_one, equation_two), equation_three)[0]
    )
    return epsilon



def linear_interpolation(
    var_converge,
    var_epsilon_zero,
    overshooting_in_epsilon_zero,
    undershooting_in_epsilon_converge,
):
    """
    Linearly interpolate between two variables.
    """
    var_converge = np.array(var_converge[0])
    N = len(var_converge)
    t = math.log10(abs(undershooting_in_epsilon_converge)) / math.log10(
        overshooting_in_epsilon_zero
    )
    var_linear_interpolation = []
    for i in range(N):
        temp = var_converge[i] + (var_epsilon_zero[i] - var_converge[i]) * t
        var_linear_interpolation.append(temp)
    return var_linear_interpolation


def optimize_linear_interpolation(self):
        """
        Calculate variable through linear interpolation.
        """
        J, J_hat = 0, 0
        x = 5
        y = 5
        z = 5
        iteration_number = 50000
        var = tf.Variable(np.array([x, y, z]).reshape(1, -1), dtype=tf.float32)
        var_prev = np.array(var[0])
        iter_epsilon_zero = 0
        iter_gradient_zero = 0
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        var_epsilon_zero = 0
        for i in range(iteration_number):
            grads = self.calculate_gradient(var, epsilon=0)
            zipped = zip([grads], [var])
            opt.apply_gradients(zipped)
            (
                equation_one,
                equation_two,
                equation_three,
            ) = FunctionOne.calculate_equation_value(var)
            if (
                equation_one < 0.0001
                or equation_two < 0.0001
                or equation_three < 0.0001
            ) and iter_epsilon_zero == 0:
                var_epsilon_zero = np.array(var[0])
                print("Epsilon zero variable", var)
                overshooting_in_epsilon_zero = calculate_epsilon_zero(var)
                print(
                    "Maximal overshooting in epsilon zero", overshooting_in_epsilon_zero
                )
                iter_epsilon_zero = 1
            if (
                float(grads[0][0]) < 0
                or float(grads[0][1]) < 0
                or float(grads[0][2]) < 0
            ) and iter_gradient_zero == 0:
                print(
                    "Gradient and variable when first gradient becomes less than zero",
                    grads,
                    var,
                )
                iter_gradient_zero = 1
        undershooting_in_epsilon_converge = calculate_epsilon_converge(var)
        var_linear_interpolation = linear_interpolation(
            var_converge=var,
            var_epsilon_zero=var_epsilon_zero,
            overshooting_in_epsilon_zero=overshooting_in_epsilon_zero,
            undershooting_in_epsilon_converge=undershooting_in_epsilon_converge,
        )
        constraint_violations = FunctionOne.calculate_constraint_violations(
            var_linear_interpolation
        )
        max_constraint_violation = min(constraint_violations)
        euclidian_distance = calculate_euclidian_distance(
            var_linear_interpolation, self.optimal_solution
        )
        for i in range(len(var_linear_interpolation)):
            J += var_linear_interpolation[i]
            J_hat += var_linear_interpolation[i]
            J_hat += self.C * constraint_violations[i] ** 2
        print("Variable linear interpolation ", var_linear_interpolation)
        print("Maximum constraint violation ", max_constraint_violation)
        print("Euclidian distance ", euclidian_distance)
        print("J_hat ", J_hat)
        print("J ", J)