import math
from src.arrayUtils import contains, ones


def integrate_trapezoid(y: list, x: list) -> float:
    """
    time complexity: O(n)
    """

    sum = 0

    for i in range(len(x) - 1):
        next_index = i + 1
        sum += (x[next_index] - x[i]) * (y[next_index] + y[i])

    return sum / 2 # moved the constant out to do only one division


def slope(x_one: float, y_one: float, x_two: float, y_two: float) -> float:
    """
    slope formula
    """

    return (y_two - y_one) / (x_two - x_one)


def linear_interpolation(interpolated_x: list, x: list, y: list) -> list:
    """
    time complexity: O(n + m)
    """

    interpolated_y = []

    if len(y) < 2 or len(x) < 2 or len(interpolated_x) == 0:
        return interpolated_y

    interpolated_x_index = 0
    i = 0

    # these vars are in this scope so thay can be used for interpolation after the interval
    x_one = x[i]
    y_one = y[i]
    x_two = x[i + 1]
    y_two = y[i + 1]
    current_slope = slope(x_one, y_one, x_two, y_two)

    # interpolate values before the interval
    while interpolated_x_index < len(interpolated_x) and interpolated_x[interpolated_x_index] < x[0]:
        interpolated_value = current_slope * (interpolated_x[interpolated_x_index] - x_one) + y_one
        interpolated_y.append(interpolated_value)
        interpolated_x_index += 1

    # interpolate values in the interval
    while i < len(x) - 1 and interpolated_x_index < len(interpolated_x):
        x_one = x[i]
        y_one = y[i]

        i += 1

        if x_one == interpolated_x[interpolated_x_index]:
            interpolated_y.append(y_one)
            interpolated_x_index += 1
        else:
            x_two = x[i]
            y_two = y[i]
            current_slope = slope(x_one, y_one, x_two, y_two)

            while interpolated_x_index < len(interpolated_x) and interpolated_x[interpolated_x_index] < x_one:
                interpolated_value = current_slope * (interpolated_x[interpolated_x_index] - x_one) + y_one
                interpolated_y.append(interpolated_value)
                interpolated_x_index += 1

    # interpolate values after the interval
    x_two = x[i]
    y_two = y[i]
    current_slope = slope(x_one, y_one, x_two, y_two)

    while interpolated_x_index < len(interpolated_x):
        interpolated_value = current_slope * (interpolated_x[interpolated_x_index] - x_one) + y_one
        interpolated_y.append(interpolated_value)
        interpolated_x_index += 1

    return interpolated_y


def linear_space(min: float, max: float, samples: int) -> list:
    """
    time complexity: O(n)

    create a list of evenly spaced points
    """

    arr = []
    sum = min
    samples -= 1
    interval_size = (max - min) / (samples)
    i = 0

    while i < samples:
        arr.append(sum)
        sum += interval_size
        i+=1

    arr.append(max)

    return arr


def vector_power(array, power):
    new_vector = []
    i = 0
    while i < len(array):
        new_vector.append(math.pow(array[i], power))
        i += 1
    return new_vector


def power_space(start, end, num_points, power:int = 2):
    space = vector_subtraction(ones(num_points), vector_power(space, power))
    # scale normalized sequence to range [start, end]
    return scalar_addition(start, scalar_multiplication(end - start, space))


def reverse_power_space(start, end, num_points, power:int = 2):
    space = linear_space(0, 1, num_points)[::-1]
    space = vector_subtraction(ones(num_points), vector_power(space, power))
    # scale normalized sequence to range [start, end]
    return scalar_addition(start, scalar_multiplication(end - start, space))


def linear_interpolation_single_point(x: float, x_one: float, y_one: float, x_two: float, y_two: float) -> float:
    return slope(x_one, y_one, x_two, y_two) * (x - x_one) + y_one


def integrate_cumulative_trapezoid(y: list, x: list, initial: float = None) -> list:
    """
    time complexity: O(n)
    """

    sums = []
    sum = 0

    if initial is not None:
        sum = initial
        sums.append(sum)

    for i in range(len(x) - 1):
        next_index = i + 1
        sum += (x[next_index] - x[i]) * (y[next_index] + y[i]) / 2
        sums.append(sum)

    return sums


def scalar_multiplication(scalar: float, array: list) -> list:
    new_vector = []
    i = 0
    while i < len(array):
        new_vector.append(array[i] * scalar)
        i += 1
    return new_vector


def scalar_division(scalar: float, array: list) -> list:
    new_vector = []
    i = 0
    while i < len(array):
        new_vector.append(array[i] / scalar)
        i += 1
    return new_vector


def vector_addition(array1: list, array2: list) -> list:
    new_vector = []
    i = 0
    while i < len(array1):
        new_vector.append(array1[i] + array2[i])
        i += 1
    return new_vector


def vector_subtraction(array1: list, array2: list) -> list:
    new_vector = []
    i = 0
    while i < len(array1):
        new_vector.append(array1[i] - array2[i])
        i += 1
    return new_vector


def vector_division(array1: list, array2: list) -> list:
    new_vector = []
    i = 0
    while i < len(array1):
        new_vector.append(array1[i] / array2[i])
        i += 1
    return new_vector


def runge_kutta_order_4_explicit_step(y_n: list, t: float, dt: float, function: callable, args:any = (), kwargs:any = {}) -> list:
    """
    explicit
    for non-stiff problems
    any order differential equation

    y_n: initial y
    t: time step
    dt: time step size
    function: function(t:float, y: list, args, kwargs) -> list

    returns: y_{n+1}
    """

    k1 = function(t, y_n, *args, **kwargs)
    k2 = function(t + dt / 2, vector_addition(y_n, scalar_multiplication(dt / 2, k1)), *args, **kwargs)
    k3 = function(t + dt / 2, vector_addition(y_n, scalar_multiplication(dt / 2, k2)), *args, **kwargs)
    k4 = function(t + dt, vector_addition(y_n, scalar_multiplication(dt, k3)), *args, **kwargs)

    k2 = vector_addition(k2, k3)
    k2 = scalar_multiplication(2, k2)

    k1 = vector_addition(k1, k2)
    k1 = vector_addition(k1, k4)
    k1 = scalar_multiplication(dt / 6, k1)

    return vector_addition(y_n, k1)


def runge_kutta_order_4_explicit(function: callable, time_span: list, y_i: list, dt_max: float = math.inf, initial_step_size: float = 0.01, dy_max: float = 0.1, dy_min:float = 0.001, args:any = (), kwargs:any = {}) -> list:
    """
    explicit
    for non-stiff problems
    adaptive step size
    any order differential equation

    dy_max: speed
    dy_min: accuracy
    dt_max: max time step size
    initial_step_size: starting step size
    time_span: time bounds
    function: function(t:float, y: list, args, kwargs) -> list

    returns: [time_points, y]
    """

    dt = initial_step_size
    t_i = time_span[0]
    t_f = time_span[1]
    time_points = [t_i]
    t = t_i
    y = [y_i]
    y_integrated = y_i

    while t < t_f:
        y_integrated_new = runge_kutta_order_4_explicit_step(y_integrated, t, dt, function, args, kwargs)

        # predicted next y
        slope_current = function(t, y_integrated, *args, **kwargs)
        y_next = vector_addition(y_integrated, scalar_multiplication(dt, slope_current))
        y_difference_1 = vector_abs(vector_subtraction(y_next, y_integrated_new))

        # predicted current y
        slope_next = function(t + dt, y_integrated_new, *args, **kwargs)
        y_previous = vector_subtraction(y_integrated_new, scalar_multiplication(dt, slope_next))
        y_difference_2 = vector_abs(vector_subtraction(y_integrated, y_previous))

        if max(y_difference_1) > dy_max or max(y_difference_2) > dy_max: # predicted y max
            dt /= 2
        else:
            y_integrated = y_integrated_new
            y.append(y_integrated)
            t += dt
            time_points.append(t)

            if min(y_difference_1) <= dy_min or min(y_difference_2) <= dy_min: # predicted y min
                dt *= 2

                if dt > dt_max:
                    dt = dt_max

    # last step
    # the loop overshoots t_f for accuracy. we then get y_{t_f} without loss of accuracy
    if time_points[-1] > t_f:
        t = time_points[-2]
        dt = t_f - t

        y[-1] = runge_kutta_order_4_explicit_step(y[-2], t, dt, function, args, kwargs)
        time_points[-1] = t_f

    return [time_points, y]


def midpoint_no_overflow(low: float, high: float) -> float:
    """
    no overflow midpoint formula
    """

    return low + (high - low) / 2


def midpoint(a: float, b: float) -> float:
    """
    midpoint formula
    """

    return (a + b) / 2


def bisection(function: callable, a: float, b: float, max_iterations: int = 1024) -> float:
    """
    root finding algorithm

    a: start of interval
    b: end of interval

    returns None if fails
    """

    f_a = function(a)
    f_b = function(b)

    if f_a * f_b >= 0:
        return None

    i = 0
    m = 0

    while i < max_iterations:
        m = midpoint(a, b)
        f_m = function(m)

        if f_m == 0:
            return m
        elif f_b * f_m < 0:
            a = m
            f_a = f_m
        elif f_a * f_m < 0:
            b = m
            f_b = f_m
        else:
            return None

        i += 1

    return m


def secant(function: callable, a: float, b: float, max_iterations: int = 1024) -> float:
    """
    root finding algorithm

    a: start of interval
    b: end of interval

    returns None if fails
    """

    f_a = function(a)
    f_b = function(b)

    if f_a * f_b >= 0:
        return None

    i = 0
    m = 0

    while i < max_iterations:
        m = a - f_a * (b - a) / (f_b - f_a)
        f_m = function(m)

        if f_m == 0:
            return m
        elif f_b * f_m < 0:
            a = m
            f_a = f_m
        elif f_a * f_m < 0:
            b = m
            f_b = f_m
        else:
            return None

        i += 1

    return m


def newtons_method(function: callable, x: float, derivative: callable = None, tolerance: float = 0.0000001, max_iterations: int = 1024) -> float:
    """
    root finding algorithm

    x: inital guess for x
    function: function(x:float) -> float
    derivative: derivative(x:float) -> float
    tolerance: accuracy of the result
    max_iterations: max iterations allowed to find the root

    returns None if fails
    """

    i = 0

    while i < max_iterations:
        f_x = function(x)

        if abs(f_x) < tolerance:
            return x

        df_x = 0

        if derivative is None:
            df_x = (function(x + tolerance) - f_x) / tolerance # Approximate derivative
        else:
            df_x = derivative(x)

        if df_x == 0:
            return None

        x -= f_x / df_x
        i += 1

    return x


def vector_abs(array: list) -> list:
    new_array = []
    i = 0

    while i < len(array):
        new_array.append(abs(array[i]))
        i += 1

    return new_array


def contains_nan_or_infinity(array: list) -> bool:
    i = 0

    while i < len(array):
        element = array[i]
        if math.isnan(element) or math.isinf(element):
            return True
        i += 1

    return False


def backward_differentiation_formula_implicit_fixed_point_iteration(function: callable, t: float, dt: float, x: list, y_previous: list, tolerance: float = 0.0000001, max_iterations: int = 1024, args:any = (), kwargs:any = {}) -> list:
    """
    x: initial x guess
    t: time step
    dt: time step size
    y_previous: the addition of all previous y's. ex: y_{n} + y_{n-1} + ...
    function: function(t:float, x: list, args, kwargs) -> list
    tolerance: accuracy of the result
    max_iterations: max iterations allowed to find the fixed point

    returns: the fixed point y_{n+1}
    """

    i = 0

    while i < max_iterations:
        y = vector_subtraction(scalar_multiplication(dt, function(t, x, *args, **kwargs)), y_previous)

        if contains_nan_or_infinity(y):
            return x

        if max(vector_abs(vector_subtraction(x, y))) < tolerance:
            return y

        x = y
        i += 1

    return x


def backward_differentiation_formula_implicit_newtons_method(function: callable, t: float, dt: float, x: list, y_previous: list, jacobian: callable = None, tolerance: float = 0.0000001, max_iterations: int = 1024, args:any = (), kwargs:any = {}) -> list:
    """
    x: initial x guess
    t: time step
    dt: time step size
    y_previous: the addition of all previous y's. ex: y_{n} + y_{n-1} + ...
    function: function(t:float, x: list, args, kwargs) -> list
    jacobian: jacobian(t:float, x: list, args, kwargs) -> list
    tolerance: accuracy of the result
    max_iterations: max iterations allowed to find the fixed point

    returns: the fixed point y_{n+1}
    """

    i = 0

    if jacobian is None:
        while i < max_iterations:
            f = function(t, x, *args, **kwargs) # for jacobian
            y = vector_subtraction(vector_addition(x, y_previous), scalar_multiplication(dt, f))

            if contains_nan_or_infinity(y) or max(vector_abs(y)) < tolerance:
                return x

            x2 = scalar_addition(tolerance, x)
            f2 = function(t, x2, *args, **kwargs)
            j = vector_subtraction(ones(len(x)), scalar_multiplication(dt / tolerance, vector_subtraction(f2, f))) # Approximate jacobian

            if contains(j, 0):
                return x

            x = vector_subtraction(x, vector_division(y, j))
            i += 1
    else:
        while i < max_iterations:
            y = vector_subtraction(vector_addition(x, y_previous), scalar_multiplication(dt, function(t, x, *args, **kwargs)))

            if contains_nan_or_infinity(y) or max(vector_abs(y)) < tolerance:
                return x

            j = vector_subtraction(ones(len(x)), scalar_multiplication(dt, jacobian(t, x, *args, **kwargs)))

            if contains(j, 0):
                return x

            x = vector_subtraction(x, vector_division(y, j))
            i += 1

    return x


def backward_differentiation_formula_order_6_implicit_fixed_point_iteration_step(y_n: list, t: float, dt: float, function: callable, tolerance: float = 0.0000001, max_iterations: int = 1024, args:any = (), kwargs:any = {}) -> list:
    """
    implicit
    for stiff problems
    any order differential equation
    https://en.wikipedia.org/wiki/Backward_differentiation_formula

    y_n: initial y
    t: time step
    dt: time step size
    function: function(t:float, y: list, args, kwargs) -> list
    tolerance: accuracy of the result
    max_iterations: max iterations allowed to find the root

    returns: y_{n+1}
    """

    dt /= 6

    y_previous = scalar_multiplication(-1, y_n)
    y_1 = backward_differentiation_formula_implicit_fixed_point_iteration(function, t + dt, dt, y_n, y_previous, tolerance, max_iterations, args, kwargs)

    t_1 = scalar_multiplication(1/3, y_n)
    t_2 = scalar_multiplication(-4/3, y_1)
    y_previous = vector_addition(t_2, t_1)
    y_2 = backward_differentiation_formula_implicit_fixed_point_iteration(function, t + dt * 2, dt * 2 / 3, y_1, y_previous, tolerance, max_iterations, args, kwargs)

    t_1 = scalar_multiplication(-2/11, y_n)
    t_2 = scalar_multiplication(9/11, y_1)
    t_3 = scalar_multiplication(-18/11, y_2)
    y_previous = vector_addition(t_2, t_1)
    y_previous = vector_addition(t_3, y_previous)
    y_3 = backward_differentiation_formula_implicit_fixed_point_iteration(function, t + dt * 3, dt * 6 / 11, y_2, y_previous, tolerance, max_iterations, args, kwargs)

    t_1 = scalar_multiplication(3/25, y_n)
    t_2 = scalar_multiplication(-16/25, y_1)
    t_3 = scalar_multiplication(36/25, y_2)
    t_4 = scalar_multiplication(-48/25, y_3)
    y_previous = vector_addition(t_2, t_1)
    y_previous = vector_addition(t_3, y_previous)
    y_previous = vector_addition(t_4, y_previous)
    y_4 = backward_differentiation_formula_implicit_fixed_point_iteration(function, t + dt * 4, dt * 12 / 25, y_3, y_previous, tolerance, max_iterations, args, kwargs)

    t_1 = scalar_multiplication(-12/137, y_n)
    t_2 = scalar_multiplication(75/137, y_1)
    t_3 = scalar_multiplication(-200/137, y_2)
    t_4 = scalar_multiplication(300/137, y_3)
    t_5 = scalar_multiplication(-300/137, y_4)
    y_previous = vector_addition(t_2, t_1)
    y_previous = vector_addition(t_3, y_previous)
    y_previous = vector_addition(t_4, y_previous)
    y_previous = vector_addition(t_5, y_previous)
    y_5 = backward_differentiation_formula_implicit_fixed_point_iteration(function, t + dt * 5, dt * 60 / 137, y_4, y_previous, tolerance, max_iterations, args, kwargs)

    t_1 = scalar_multiplication(10/147, y_n)
    t_2 = scalar_multiplication(-72/147, y_1)
    t_3 = scalar_multiplication(225/147, y_2)
    t_4 = scalar_multiplication(-400/147, y_3)
    t_5 = scalar_multiplication(450/147, y_4)
    t_6 = scalar_multiplication(-360/147, y_5)
    y_previous = vector_addition(t_2, t_1)
    y_previous = vector_addition(t_3, y_previous)
    y_previous = vector_addition(t_4, y_previous)
    y_previous = vector_addition(t_5, y_previous)
    y_previous = vector_addition(t_6, y_previous)
    y_6 = backward_differentiation_formula_implicit_fixed_point_iteration(function, t + dt * 6, dt * 60 / 147, y_5, y_previous, tolerance, max_iterations, args, kwargs)

    return y_6


def backward_differentiation_formula_order_6_implicit_newtons_method_step(y_n: list, t: float, dt: float, function: callable, jacobian: callable = None, tolerance: float = 0.0000001, max_iterations: int = 1024, args:any = (), kwargs:any = {}) -> list:
    """
    implicit
    for stiff problems
    any order differential equation
    https://en.wikipedia.org/wiki/Backward_differentiation_formula

    y_n: initial y
    t: time step
    dt: time step size
    function: function(t:float, y: list, args, kwargs) -> list
    jacobian: jacobian(t:float, x: list, args, kwargs) -> list
    tolerance: accuracy of the result
    max_iterations: max iterations allowed to find the root

    returns: y_{n+1}
    """

    dt /= 6

    y_previous = scalar_multiplication(-1, y_n)
    y_1 = backward_differentiation_formula_implicit_newtons_method(function, t + dt, dt, y_n, y_previous, jacobian, tolerance, max_iterations, args, kwargs)

    t_1 = scalar_multiplication(1/3, y_n)
    t_2 = scalar_multiplication(-4/3, y_1)
    y_previous = vector_addition(t_2, t_1)
    y_2 = backward_differentiation_formula_implicit_newtons_method(function, t + dt * 2, dt * 2 / 3, y_1, y_previous, jacobian, tolerance, max_iterations, args, kwargs)

    t_1 = scalar_multiplication(-2/11, y_n)
    t_2 = scalar_multiplication(9/11, y_1)
    t_3 = scalar_multiplication(-18/11, y_2)
    y_previous = vector_addition(t_2, t_1)
    y_previous = vector_addition(t_3, y_previous)
    y_3 = backward_differentiation_formula_implicit_newtons_method(function, t + dt * 3, dt * 6 / 11, y_2, y_previous, jacobian, tolerance, max_iterations, args, kwargs)

    t_1 = scalar_multiplication(3/25, y_n)
    t_2 = scalar_multiplication(-16/25, y_1)
    t_3 = scalar_multiplication(36/25, y_2)
    t_4 = scalar_multiplication(-48/25, y_3)
    y_previous = vector_addition(t_2, t_1)
    y_previous = vector_addition(t_3, y_previous)
    y_previous = vector_addition(t_4, y_previous)
    y_4 = backward_differentiation_formula_implicit_newtons_method(function, t + dt * 4, dt * 12 / 25, y_3, y_previous, jacobian, tolerance, max_iterations, args, kwargs)

    t_1 = scalar_multiplication(-12/137, y_n)
    t_2 = scalar_multiplication(75/137, y_1)
    t_3 = scalar_multiplication(-200/137, y_2)
    t_4 = scalar_multiplication(300/137, y_3)
    t_5 = scalar_multiplication(-300/137, y_4)
    y_previous = vector_addition(t_2, t_1)
    y_previous = vector_addition(t_3, y_previous)
    y_previous = vector_addition(t_4, y_previous)
    y_previous = vector_addition(t_5, y_previous)
    y_5 = backward_differentiation_formula_implicit_newtons_method(function, t + dt * 5, dt * 60 / 137, y_4, y_previous, jacobian, tolerance, max_iterations, args, kwargs)

    t_1 = scalar_multiplication(10/147, y_n)
    t_2 = scalar_multiplication(-72/147, y_1)
    t_3 = scalar_multiplication(225/147, y_2)
    t_4 = scalar_multiplication(-400/147, y_3)
    t_5 = scalar_multiplication(450/147, y_4)
    t_6 = scalar_multiplication(-360/147, y_5)
    y_previous = vector_addition(t_2, t_1)
    y_previous = vector_addition(t_3, y_previous)
    y_previous = vector_addition(t_4, y_previous)
    y_previous = vector_addition(t_5, y_previous)
    y_previous = vector_addition(t_6, y_previous)
    y_6 = backward_differentiation_formula_implicit_newtons_method(function, t + dt * 6, dt * 60 / 147, y_5, y_previous, jacobian, tolerance, max_iterations, args, kwargs)

    return y_6


def scalar_addition(scalar: float, array: list) -> list:
    new_vector = []
    i = 0
    while i < len(array):
        new_vector.append(array[i] + scalar)
        i += 1
    return new_vector


def backward_differentiation_formula_order_6_implicit_fixed_point_iteration(function: callable, time_span: list, y_i: list, dt_max: float = math.inf, initial_step_size: float = 0.01, dy_max: float = 0.1, dy_min:float = 0.001, root_finding_tolerance: float = 0.0000001, root_finding_max_iterations: int = 1024, args:any = (), kwargs:any = {}) -> list:
    """
    implicit
    for stiff problems
    adaptive step size
    any order differential equation

    dy_max: speed
    dy_min: accuracy
    dt_max: max time step size
    initial_step_size: starting step size
    time_span: time bounds
    function: function(t:float, y: list, args, kwargs) -> list
    root_finding_tolerance: accuracy of the step result
    root_finding_max_iterations: max iterations allowed to find the root in a step

    returns: [time_points, y]
    """

    dt = initial_step_size
    t_i = time_span[0]
    t_f = time_span[1]
    time_points = [t_i]
    t = t_i
    y = [y_i]
    y_integrated = y_i

    while t < t_f:
        y_integrated_new = backward_differentiation_formula_order_6_implicit_fixed_point_iteration_step(y_integrated, t, dt, function, root_finding_tolerance, root_finding_max_iterations, args, kwargs)

        # predicted next y
        slope_current = function(t, y_integrated, *args, **kwargs)
        y_next = vector_addition(y_integrated, scalar_multiplication(dt, slope_current))
        y_difference_1 = vector_abs(vector_subtraction(y_next, y_integrated_new))

        # predicted current y
        slope_next = function(t + dt, y_integrated_new, *args, **kwargs)
        y_previous = vector_subtraction(y_integrated_new, scalar_multiplication(dt, slope_next))
        y_difference_2 = vector_abs(vector_subtraction(y_integrated, y_previous))

        if max(y_difference_1) > dy_max or max(y_difference_2) > dy_max: # predicted y max
            dt /= 2
        else:
            y_integrated = y_integrated_new
            y.append(y_integrated)
            t += dt
            time_points.append(t)

            if min(y_difference_1) <= dy_min or min(y_difference_2) <= dy_min: # predicted y min
                dt *= 2

                if dt > dt_max:
                    dt = dt_max

    # last step
    # the loop overshoots t_f for accuracy. we then get y_{t_f} without loss of accuracy
    if time_points[-1] > t_f:
        t = time_points[-2]
        dt = t_f - t

        y[-1] = backward_differentiation_formula_order_6_implicit_fixed_point_iteration_step(y[-2], t, dt, function, root_finding_tolerance, root_finding_max_iterations, args, kwargs)
        time_points[-1] = t_f

    return [time_points, y]


def backward_differentiation_formula_order_6_implicit_newtons_method(function: callable, time_span: list, y_i: list, dt_max: float = math.inf, initial_step_size: float = 0.01, dy_max: float = 0.1, dy_min:float = 0.001, jacobian: callable = None, root_finding_tolerance: float = 0.0000001, root_finding_max_iterations: int = 1024, args:any = (), kwargs:any = {}) -> list:
    """
    implicit
    for stiff problems
    adaptive step size
    any order differential equation

    dy_max: speed
    dy_min: accuracy
    dt_max: max time step size
    initial_step_size: starting step size
    time_span: time bounds
    function: function(t:float, y: list, args, kwargs) -> list
    jacobian: jacobian(t:float, x: list, args, kwargs) -> list
    root_finding_tolerance: accuracy of the step result
    root_finding_max_iterations: max iterations allowed to find the root in a step

    https://www.uni-muenster.de/imperia/md/content/physik_tp/lectures/ss2017/numerische_Methoden_fuer_komplexe_Systeme_II/rkm-1.pdf

    returns: [time_points, y]
    """

    dt = initial_step_size
    t_i = time_span[0]
    t_f = time_span[1]
    time_points = [t_i]
    t = t_i
    y = [y_i]
    y_integrated = y_i

    order = 6
    safety_factor = 0.9

    while t < t_f:
        y_integrated_new = backward_differentiation_formula_order_6_implicit_newtons_method_step(y_integrated, t, dt, function, jacobian, root_finding_tolerance, root_finding_max_iterations, args, kwargs)

        # predicted next y
        slope_current = function(t, y_integrated, *args, **kwargs)
        y_next = vector_addition(y_integrated, scalar_multiplication(dt, slope_current))
        y_difference_1 = vector_abs(vector_subtraction(y_next, y_integrated_new))

        # predicted current y
        slope_next = function(t + dt, y_integrated_new, *args, **kwargs)
        y_previous = vector_subtraction(y_integrated_new, scalar_multiplication(dt, slope_next))
        y_difference_2 = vector_abs(vector_subtraction(y_integrated, y_previous))

        maximum = max(max(y_difference_1), max(y_difference_2))
        if maximum > dy_max: # predicted y max
            if math.isnan(maximum) or math.isinf(maximum):
                dt /= 2
            else:
                dt *= safety_factor * math.pow(dy_max / maximum, 1 / (order + 1))
        else:
            y_integrated = y_integrated_new
            y.append(y_integrated)
            t += dt
            time_points.append(t)

            minimum = min(min(y_difference_1), min(y_difference_2))
            if minimum <= dy_min: # predicted y min
                if minimum == 0:
                    dt *= 2
                else:
                    dt *= safety_factor * math.pow(dy_min / minimum, 1 / order)

                if dt > dt_max:
                    dt = dt_max

    # last step
    # the loop overshoots t_f for accuracy. we then get y_{t_f} without loss of accuracy
    if time_points[-1] > t_f:
        t = time_points[-2]
        dt = t_f - t

        y[-1] = backward_differentiation_formula_order_6_implicit_newtons_method_step(y[-2], t, dt, function, jacobian, root_finding_tolerance, root_finding_max_iterations, args, kwargs)
        time_points[-1] = t_f

    return [time_points, y]


def generate_multinomial_terms_recursion_call(number_of_variables, terms, remaining_degree, current_term, current_variable_index):
    # base case: all variables are now used
    if current_variable_index == number_of_variables:
        if remaining_degree == 0: # reject terms that don't add up to n (remaining_degree)
            terms.append(current_term)
        return

    # go through the next exponents for the next variable
    current_variable_index += 1
    next_exponent = 0
    while next_exponent < remaining_degree + 1:
        new_current_term_part = current_term.copy()
        new_current_term_part.append(next_exponent)
        new_remaining_degree = remaining_degree - next_exponent
        generate_multinomial_terms_recursion_call(number_of_variables, terms, new_remaining_degree, new_current_term_part, current_variable_index)
        next_exponent += 1


def generate_multinomial_terms_recursion(number_of_variables, degree):
    terms = []
    generate_multinomial_terms_recursion_call(number_of_variables, terms, degree, [], 0)
    return terms


def generate_multinomial_terms(number_of_variables, degree):
    terms = [[]]
    final_terms = []

    i = 0
    while i < number_of_variables:
        new_terms = []

        j = 0
        while j < len(terms):
            term = terms[j]
            term_sum = sum(term)

            k = 0
            while k <= degree:
                current_term_sum = term_sum + k

                if current_term_sum <= degree: # filter out combinations > degree
                    new_term = term.copy()
                    new_term.append(k)

                    if i + 1 == number_of_variables and current_term_sum == degree: # found a term
                        final_terms.append(new_term)
                    else:
                        new_terms.append(new_term) # build upon this term

                k += 1

            j += 1

        terms = new_terms
        i += 1

    return final_terms
