import numpy as np
from typing import Union


# random_range DEFINE
class RANDOM_RANGE:
    def __init__(
        self,
        r_c_size: tuple[int, int] = (1, 20),
        alpha_f: tuple = (1, 20),
        beta_c_l: tuple = (1, 20),
        u_s: tuple[int, int] = (2, 20),
        phi_s_l: tuple[int, int] = (1, 10),
    ):
        self.r_c_size = r_c_size
        self.alpha_f = alpha_f
        self.beta_c_l = beta_c_l
        self.u_s = u_s
        self.phi_s_l = phi_s_l


# SET
class SIZE:
    def __init__(
        self,
        c_size: int = 10,
        f_size: int = 10,
        m_size: int = 10,
        r_c_size_list: np.ndarray = None,
        l_size: int = 10,
        s_size: int = 10,
        seed: int = 1,
        random_range: RANDOM_RANGE = None,
    ):
        """
        :param c_size: the size of the set of aircraft
        :param f_size: the size of the set of flights
        :param m_size: the size of the set of planned maintenances
        :param r_c_size_list: the size of the set of AOGs of aircraft c belongs C
        :param l_size: the size of the set of routes
        :param s_size: the size of the set of departure/arrival slots
        """
        if random_range is None:
            random_range = RANDOM_RANGE()
        np.random.seed(seed)
        self.c_size = c_size
        self.f_size = f_size
        self.m_size = m_size
        if r_c_size_list is None:
            self.r_c_size_list = np.random.randint(
                random_range.r_c_size[0], random_range.r_c_size[1], size=c_size
            )
        else:
            self.r_c_size_list = r_c_size_list
        self.l_size = l_size
        self.s_size = s_size
        self.random_range = random_range
        self.seed = seed

    def __str__(self):
        return "SIZE(c_size={}, f_size={}, m_size={}, r_c_size_list={}, l_size={}, s_size={})".format(
            self.c_size,
            self.f_size,
            self.m_size,
            self.r_c_size_list.tolist(),
            self.l_size,
            self.s_size,
        )


class PARAMETER:
    def __init__(
        self,
        alpha_f_array: Union[np.ndarray, str, None] = None,
        beta_c_l_array: Union[np.ndarray, str, None] = None,
        delta_f_l_array: Union[np.ndarray, str, None] = None,
        delta_m_l_array: Union[np.ndarray, str, None] = None,
        delta_Rc_r_l_array: Union[list, str, None] = None,
        u_s_array: Union[np.ndarray, str, None] = None,
        phi_s_l_array: Union[np.ndarray, str, None] = None,
        size: SIZE = None,
    ):
        """
        None: zeros

        "random": random parameter

        np.ndarray: assigned parameter
        """
        if size is None:
            size = SIZE()
        np.random.seed(size.seed)
        random_range = size.random_range
        if alpha_f_array is None:
            self.alpha_f_array = np.zeros(size.f_size)
        elif isinstance(alpha_f_array, str):
            self.alpha_f_array = np.random.randint(
                random_range.alpha_f[0], random_range.alpha_f[1], size=size.f_size
            )
        else:
            self.alpha_f_array = alpha_f_array

        if beta_c_l_array is None:
            self.beta_c_l_array = np.zeros((size.c_size, size.l_size))
        elif isinstance(beta_c_l_array, str):
            self.beta_c_l_array = np.random.randint(
                random_range.beta_c_l[0],
                random_range.beta_c_l[1],
                size=(size.c_size, size.l_size),
            )
        else:
            self.beta_c_l_array = beta_c_l_array

        if delta_f_l_array is None:
            self.delta_f_l_array = np.zeros((size.f_size, size.l_size))
        elif isinstance(delta_f_l_array, str):
            self.delta_f_l_array = np.random.randint(
                0, 2, size=(size.f_size, size.l_size)
            )
        else:
            self.delta_f_l_array = delta_f_l_array

        if delta_m_l_array is None:
            self.delta_m_l_array = np.zeros((size.m_size, size.l_size))
        elif isinstance(delta_m_l_array, str):
            self.delta_m_l_array = np.random.randint(
                0, 2, size=(size.m_size, size.l_size)
            )
        else:
            self.delta_m_l_array = delta_m_l_array

        if delta_Rc_r_l_array is None:
            self.delta_Rc_r_l_array = [
                np.zeros((R_C_SIZE, size.l_size)) for R_C_SIZE in size.r_c_size_list
            ]
        elif isinstance(delta_Rc_r_l_array, str):
            self.delta_Rc_r_l_array = [
                np.random.randint(0, 2, size=(R_C_SIZE, size.l_size))
                for R_C_SIZE in size.r_c_size_list
            ]
        else:
            self.delta_Rc_r_l_array = delta_Rc_r_l_array

        if u_s_array is None:
            self.u_s_array = np.zeros(size.s_size)
        elif isinstance(u_s_array, str):
            self.u_s_array = np.random.randint(
                random_range.u_s[0], random_range.u_s[1], size=size.s_size
            )
        else:
            self.u_s_array = u_s_array

        if phi_s_l_array is None:
            self.phi_s_l_array = [np.zeros(size.l_size) for _ in range(size.s_size)]
        elif isinstance(phi_s_l_array, str):
            self.phi_s_l_array = np.array(
                [
                    np.random.randint(
                        random_range.phi_s_l[0],
                        random_range.phi_s_l[1],
                        size=size.l_size,
                    )
                    for _ in range(size.s_size)
                ]
            )
        else:
            self.phi_s_l_array = phi_s_l_array

    def get_size(self):
        c_size = self.beta_c_l_array.shape[0]
        f_size = self.delta_f_l_array.shape[0]
        m_size = self.delta_m_l_array.shape[0]
        r_c_size_list = np.array([x.shape[0] for x in self.delta_Rc_r_l_array])
        l_size = self.beta_c_l_array.shape[1]
        s_size = self.u_s_array.shape[0]
        return SIZE(
            c_size=c_size,
            f_size=f_size,
            m_size=m_size,
            r_c_size_list=r_c_size_list,
            l_size=l_size,
            s_size=s_size,
        )


if __name__ == "__main__":
    parameter = PARAMETER(
        alpha_f_array="random",
        delta_f_l_array="random",
        delta_m_l_array="random",
        delta_Rc_r_l_array="random",
        u_s_array="random",
        phi_s_l_array="random",
    )
