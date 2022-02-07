import numpy as np


class Edge:
    def __init__(self, i, j, l, r):
        """

        :param i:
        :param j:
        :param l: left - neighbor
        :param r: right - neighbor
        """
        self.i = i
        self.j = j
        self.l = l
        self.r = r
        self.is_on_boundary = self.r is None

        if not self.is_on_boundary:
            self.g = np.zeros(shape=(8, 4), dtype=np.float32)
        else:
            self.g = np.zeros(shape=(6, 4), dtype=np.float32)

        self.t = np.zeros(shape=(2, 8), dtype=np.float32)
        self.h = np.zeros(shape=(2, 8), dtype=np.float32)

    def calculate_g(self, V):
        """

        :param V: Vertices
        :return:
        """
        v_i = V[self.i]
        v_j = V[self.j]
        v_l = V[self.l]

        # i
        self.g[0, :] = [v_i[0], v_i[1], 1, 0]
        self.g[1, :] = [v_i[1], -v_i[0], 0, 1]
        # j
        self.g[2, :] = [v_j[0], v_j[1], 1, 0]
        self.g[3, :] = [v_j[1], -v_j[0], 0, 1]
        # l
        self.g[4, :] = [v_l[0], v_l[1], 1, 0]
        self.g[5, :] = [v_l[1], -v_l[0], 0, 1]

        if not self.is_on_boundary:
            v_r = V[self.r]

            # r
            self.g[6, :] = [v_r[0], v_r[1], 1, 0]
            self.g[7, :] = [v_r[1], -v_r[0], 0, 1]

    def calculate_h(self, V):
        """
        :param V: Vertices
        :return:
        """
        g = self.g

        e = V[self.j] - V[self.i]

        # term 2
        term_2 = np.array([
            [e[0], e[1]],
            [e[1], -e[0]]
        ])
        self.t = (np.linalg.inv(g.T @ g) @ g.T)[:2, :]

        # term 1
        if not self.is_on_boundary:
            term_1 = np.zeros(shape=(2, 8), dtype=np.float32)
            term_2 = term_2 @ self.t
        else:
            term_1 = np.zeros(shape=(2, 6), dtype=np.float32)
            term_2 = term_2 @ self.t[:, :6]

        term_1[0, 0] = -1.
        term_1[0, 2] = 1.
        term_1[1, 1] = -1
        term_1[1, 3] = 1

        h = term_1 - term_2

        if not self.is_on_boundary:
            self.h = h
        else:
            self.h[:, :6] = h


class Solver:
    def __init__(self, edge_list, constraint_v_ids, constraint_v_coords, V, w=10.):
        self.V = V
        self.edge_list = edge_list
        self.constraint_v_ids = constraint_v_ids
        self.constraint_v_coords = constraint_v_coords
        self.w = w

    def step_one(self):
        """

        :return:
        """
        edge_list = self.edge_list
        constraint_v_coords = self.constraint_v_coords
        constraint_v_ids = self.constraint_v_ids
        w = self.w
        V = self.V

        n_E = len(edge_list)
        n_C = len(constraint_v_coords)
        n_V = len(self.V)

        H = np.zeros(shape=(2 * n_E + 2 * n_C, 2 * n_V), dtype=np.float64)
        b = np.zeros(shape=(2 * n_E + 2 * n_C, 1), dtype=np.float64)
        for k, edge in enumerate(edge_list):
            i, j, l, r = edge.i, edge.j, edge.l, edge.r if (not edge.is_on_boundary) else None
            for v_i, v in enumerate([i, j, l, r]):
                if v is None:
                    continue

                H[k * 2, v * 2] = edge.h[0, v_i * 2]
                H[k * 2 + 1, v * 2] = edge.h[1, v_i * 2]
                H[k * 2, v * 2 + 1] = edge.h[0, v_i * 2 + 1]
                H[k * 2 + 1, v * 2 + 1] = edge.h[1, v_i * 2 + 1]

        ofs = 2 * n_E
        for c_i, v_i, v_coord in zip(range(n_C), constraint_v_ids, constraint_v_coords):
            H[ofs + 2 * c_i, 2 * v_i] = w
            H[ofs + 2 * c_i + 1, 2 * v_i + 1] = w

            b[ofs + 2 * c_i, 0] = w * v_coord[0]
            b[ofs + 2 * c_i + 1, 0] = w * v_coord[1]

        #
        result = np.linalg.inv(H.T @ H) @ H.T @ b
        v_new = np.zeros(shape=(len(V), 2), dtype=np.float64)
        v_new[:, 0] = result[0::2, 0]
        v_new[:, 1] = result[1::2, 0]

        return v_new

    def calculate_T(self, V, edge):
        """

        :param V:
        :param edge:
        :return:
        """

        v_i = V[edge.i]
        v_j = V[edge.j]
        v_l = V[edge.l]

        if not edge.is_on_boundary:
            v_r = V[edge.r]
            v_ = np.stack([v_i, v_j, v_l, v_r], axis=0).reshape(-1,).astype(np.float32)
        else:
            v_ = np.stack([v_i, v_j, v_l], axis=0).reshape(-1,).astype(np.float32)

        c, s = edge.t @ v_
        T = np.array([
            [c, s],
            [-s, c]
        ]) / (c * c + s * s)

        return T

    def step_two(self, V_step1):
        """
        :return:
        """
        V = self.V
        w = self.w
        edge_list = self.edge_list
        constraint_v_ids = self.constraint_v_ids
        constraint_v_coords = self.constraint_v_coords

        n_E = len(edge_list)
        n_C = len(constraint_v_coords)
        n_V = len(self.V)
        A = np.zeros(shape=(2 * n_E + 2 * n_C, 2 * n_V), dtype=np.float32)
        b = np.zeros(shape=(2 * n_E + 2 * n_C, 1), dtype=np.float32)

        for k, edge in enumerate(edge_list):
            T_k = self.calculate_T(V_step1, edge)
            e_k = (V[edge.j] - V[edge.i]).reshape(2,)
            e_k = T_k @ e_k

            b[2 * k, 0] = e_k[0]
            b[2 * k + 1, 0] = e_k[1]
            A[2 * k, 2 * edge.j] = 1
            A[2 * k + 1, 2 * edge.j + 1] = 1
            A[2 * k, 2 * edge.i] = -1
            A[2 * k + 1, 2 * edge.i + 1] = -1

        ofs = 2 * n_E
        for c_i, v_i, v_coord in zip(range(n_C), constraint_v_ids, constraint_v_coords):
            A[ofs + 2 * c_i, 2 * v_i] = w
            A[ofs + 2 * c_i + 1, 2 * v_i + 1] = w

            b[ofs + 2 * c_i, 0] = w * v_coord[0]
            b[ofs + 2 * c_i + 1, 0] = w * v_coord[1]

        result = np.linalg.inv(A.T @ A) @ A.T @ b
        v_new2 = np.zeros(shape=(len(V), 2), dtype=np.float32)
        v_new2[:, 0] = result[0::2, 0]
        v_new2[:, 1] = result[1::2, 0]

        return v_new2


if __name__ == '__main__':
    pass
