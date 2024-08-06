from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, LpBinary
import numpy as np
from object_define import RANDOM_RANGE, SIZE, PARAMETER


class MasterProblem:
    def __init__(self, problem_name, parameter: PARAMETER, relaxation: bool = True):
        """
        :param problem_name: snake case
        :param parameter: object_define.parameter
        """
        size = parameter.get_size()
        self.problem = LpProblem(f"minimize_cost_{problem_name}", LpMinimize)
        self.__x = None
        self.__y = None
        self.x_size = (size.c_size, size.l_size)
        self.y_size = size.f_size
        self.solved = False
        self.relaxation = relaxation
        self.__set_decision_variables(size)
        self.__set_objective_function(size, parameter)
        self.__set_constraint_2(size, parameter)
        self.__set_constraint_3(size, parameter)
        self.__set_constraint_4(size, parameter)
        self.__set_constraint_5(size)
        self.__set_constraint_6(size, parameter)

    def __set_decision_variables(self, size: SIZE):
        x = np.zeros((size.c_size, size.l_size)).tolist()
        for c_idx in range(size.c_size):
            for l_idx in range(size.l_size):
                if self.relaxation:
                    x[c_idx][l_idx] = LpVariable(f"x_{c_idx}_{l_idx}", lowBound=0)
                else:
                    x[c_idx][l_idx] = LpVariable(f"x_{c_idx}_{l_idx}", cat=LpBinary)

        y = np.zeros(size.f_size).tolist()
        for f_idx in range(size.f_size):
            if self.relaxation:
                y[f_idx] = LpVariable(f"y_{f_idx}", lowBound=0)
            else:
                y[f_idx] = LpVariable(f"y_{f_idx}", cat=LpBinary)

        self.__x = x
        self.__y = y

    def __set_objective_function(self, size: SIZE, parameter: PARAMETER):
        self.problem += lpSum(
            [
                parameter.alpha_f_array[f_idx] * self.__y[f_idx]
                for f_idx in range(size.f_size)
            ]
        ) + lpSum(
            [
                parameter.beta_c_l_array[c_idx][l_idx] * self.__x[c_idx][l_idx]
                for c_idx in range(size.c_size)
                for l_idx in range(size.l_size)
            ]
        )

    def __set_constraint_2(self, size: SIZE, parameter: PARAMETER):
        for f_idx in range(size.f_size):
            self.problem += (
                lpSum(
                    [
                        parameter.delta_f_l_array[f_idx][l_idx] * self.__x[c_idx][l_idx]
                        for c_idx in range(size.c_size)
                        for l_idx in range(size.l_size)
                    ]
                )
                + self.__y[f_idx]
                == 1,
                f"Constraint_2_{f_idx}",
            )

    def __set_constraint_3(self, size: SIZE, parameter: PARAMETER):
        for m_idx in range(size.m_size):
            self.problem += (
                lpSum(
                    [
                        parameter.delta_m_l_array[m_idx][l_idx] * self.__x[c_idx][l_idx]
                        for c_idx in range(size.c_size)
                        for l_idx in range(size.l_size)
                    ]
                )
                <= 1,
                f"Constraint_3_{m_idx}",
            )

    def __set_constraint_4(self, size: SIZE, parameter: PARAMETER):
        for c_idx, R_C_SIZE in zip(range(size.c_size), size.r_c_size_list):
            for r_idx in range(R_C_SIZE):
                self.problem += (
                    lpSum(
                        [
                            parameter.delta_Rc_r_l_array[c_idx][r_idx][l_idx]
                            * self.__x[c_idx][l_idx]
                            for l_idx in range(size.l_size)
                        ]
                    )
                    == 1,
                    f"Constraint_4_{c_idx}_{r_idx}",
                )

    def __set_constraint_5(self, size: SIZE):
        for c_idx in range(size.c_size):
            self.problem += (
                lpSum([self.__x[c_idx][l_idx] for l_idx in range(size.l_size)]) <= 1,
                f"Constraint_5_{c_idx}",
            )

    def __set_constraint_6(self, size: SIZE, parameter: PARAMETER):
        for s_idx in range(size.s_size):
            self.problem += (
                lpSum(
                    [
                        parameter.phi_s_l_array[s_idx][l_idx] * self.__x[c_idx][l_idx]
                        for c_idx in range(size.c_size)
                        for l_idx in range(size.l_size)
                    ]
                )
                <= parameter.u_s_array[s_idx],
                f"Constraint_6_{s_idx}",
            )

    def solve(self, solver=None, msg=False) -> bool:
        if solver is None:
            solver = PULP_CBC_CMD(msg=msg)
        self.problem.solve(solver)
        self.solved = self.problem.sol_status == 1
        return self.solved

    def get_decision_variables_x(self):
        if not self.solved:
            print("problem unsolved, return x None")
            return None
        x_value = np.zeros(self.x_size)
        for row_idx in range(self.x_size[0]):
            for col_idx in range(self.x_size[1]):
                x_value[row_idx][col_idx] = self.__x[row_idx][col_idx].varValue
        return x_value

    def get_decision_variables_y(self):
        if not self.solved:
            print("problem unsolved, return y None")
            return None
        y_value = np.zeros(self.y_size)
        for y_idx in range(self.y_size):
            y_value[y_idx] = self.__y[y_idx].varValue
        return y_value


def get_test_param():
    alpha_f_array = np.array([9, 9, 7, 3, 9, 8])
    beta_c_l_array = np.array(
        [
            [3, 2, 6, 5, 5, 6, 8, 4, 7],
            [5, 4, 8, 7, 2, 4, 6, 9, 5],
            [7, 4, 3, 1, 5, 3, 5, 2, 8],
        ]
    )
    delta_f_l_array = np.array(
        [
            [1, 0, 0, 0, 1, 0, 1, 1, 1],
            [0, 1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 1, 0, 1, 1, 1],
        ]
    )
    delta_m_l_array = np.array(
        [
            [1, 0, 0, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 1],
        ]
    )
    delta_Rc_r_l_array = [
        np.array([[1, 1, 1, 0, 1, 0, 1, 1, 1]]),
        np.array([[1, 0, 1, 0, 1, 1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0, 1]]),
        np.array([[1, 0, 0, 1, 1, 1, 1, 0, 1]]),
    ]
    u_s_array = np.array([10, 17, 11])
    phi_s_l_array = np.array(
        [
            [1, 1, 3, 4, 2, 4, 3, 3, 3],
            [4, 1, 4, 1, 2, 4, 3, 3, 2],
            [3, 2, 4, 4, 2, 1, 4, 2, 1],
        ]
    )
    return PARAMETER(
        alpha_f_array=alpha_f_array,
        beta_c_l_array=beta_c_l_array,
        delta_f_l_array=delta_f_l_array,
        delta_m_l_array=delta_m_l_array,
        delta_Rc_r_l_array=delta_Rc_r_l_array,
        u_s_array=u_s_array,
        phi_s_l_array=phi_s_l_array,
    )


if __name__ == "__main__":
    param = get_test_param()
    master_problem_test = MasterProblem("test", param)
    master_problem_test.solve()
    for name, constraint in master_problem_test.problem.constraints.items():
        print(f"pi {name}: {constraint.pi}")
