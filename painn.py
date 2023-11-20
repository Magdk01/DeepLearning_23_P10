import torch
import torch.nn as nn
import copy
import numpy as np


def hadamard(m1, m2):
    # assert m1.shape == m2.shape
    return m1 * m2


class painn(nn.Module):
    def __init__(self, r_cut=2, n=20) -> None:
        super(painn, self).__init__()
        self.device = "cuda:0"
        self.r_cut = r_cut
        self.n = n
        self.embedding_layer = nn.Sequential(nn.Embedding(10, 128))
        self.message_model = message(self.r_cut, self.n)
        self.update_model = update()

        self.output_layers = nn.Sequential(
            nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128)
        )

    def forward(self, atomic_numbers, positional_encodings):
        self.positions = torch.tensor(positional_encodings)

        adj_list = positional_adjacency(self.positions, self.r_cut)
        self.idx_i = adj_list[0]
        self.idx_j = adj_list[1]
        self.atomic = torch.tensor(atomic_numbers)

        self.r = torch.tensor(
            r_ij_calc(adj_list, positional_encodings), dtype=torch.float32
        )
        self.s = self.embedding_layer(self.atomic).unsqueeze(1)

        # Then scaling to match all connections in molecule
        self.v = torch.zeros((self.s.shape[0], 3, 128))

        for _ in range(3):
            v, s = self.v.detach().clone(), self.s.detach().clone()

            self.v, self.s = self.message_model(
                self.s[self.idx_j], self.r, v[self.idx_j], self.idx_i
            )

            self.v = self.v + v
            self.s = self.s + s

            v, s = self.v.detach().clone(), self.s.detach().clone()

            self.v, self.s = self.update_model(s, v)

            self.v = self.v + v
            self.s = self.s + s

        out = self.output_layers(self.s)
        out = torch.sum(out)

        return out


class message(nn.Module):
    def __init__(self, r_cut, n) -> None:
        super(message, self).__init__()
        self.ø = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 384),
        )
        self.r_cut = r_cut
        self.n = n

    def forward(self, s, r, v, idx_i):
        # s-block
        ø_out = self.ø(s)
        org_r = r.detach().clone()
        assert ø_out.size(2) == 384

        # left r-block
        r = self.__rbf(r, self.n)
        r = self.__fcut(r, self.r_cut)
        w = nn.Linear(20, 384)(r)
        # w = self.__fcut(r, self.r_cut)

        assert w.size(2) == 384

        split = hadamard(w, ø_out)
        split_tensor = torch.split(split, 128, dim=2)

        # out_s = torch.sum(split_tensor[1], axis=0)
        out_s = torch.zeros(len(set(idx_i)), 1, 128)
        for idx, i in enumerate(idx_i):
            out_s[i] += split_tensor[1][idx]

        # right r-block
        org_r = org_r / torch.norm(org_r)
        # org_r = torch.norm(org_r, dim=1).unsqueeze(1).unsqueeze(2).repeat(1, 1, 128)
        org_r = org_r.unsqueeze(2).repeat(1, 1, 128)
        org_r = hadamard(split_tensor[2], org_r)  # To fix di

        # v-block
        v = hadamard(split_tensor[0], v)
        v = torch.add(org_r, v)
        out_v = torch.zeros(len(set(idx_i)), 3, 128)
        for idx, i in enumerate(idx_i):
            out_v[i] += v[idx]
        # out_v = torch.sum(v, axis=0)

        return out_v, out_s

    def __rbf(self, input, n):
        n_values = torch.arange(1, n + 1).float()  # Shape: (n,)
        r_norms = torch.norm(input, dim=1, keepdim=True)  # Shape: (12,)

        # Broadcasting r_norms to (12, n) and n_values to (12, n)
        r_norms = r_norms.unsqueeze(2).expand(-1, -1, n)

        n_values = n_values.unsqueeze(0).expand(r_norms.shape[0], 1, -1)

        res = torch.sin((n_values * torch.pi) / self.r_cut * r_norms) / r_norms

        return res

    def __fcut(self, tensor, R_c):
        # Applying the cosine cutoff function element-wise
        cos_cutoff = torch.where(
            tensor <= R_c,
            0.5 * (torch.cos(torch.pi * tensor / R_c) + 1),
            torch.zeros_like(tensor),
        )
        return cos_cutoff


class update(nn.Module):
    def __init__(self) -> None:
        super(update, self).__init__()

        self.a = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 384))

        self.V = nn.Sequential(nn.Linear(128, 128, bias=True))

        self.U = nn.Sequential(nn.Linear(128, 128, bias=True))

    def forward(self, s, v):
        # top v-block
        v = self.V(v)

        u = self.U(v)

        # s-block
        s_stack = torch.cat((torch.norm(v, dim=1, keepdim=True), s), axis=2)
        split = self.a(s_stack)
        split_tensor = torch.split(split, 128, dim=2)
        # left v-block continues
        out_v = hadamard(u, split_tensor[0])

        # right v-block continues
        # s = torch.tensordot(v, u, dims=len(v))
        s = torch.sum((u * v), dim=1).unsqueeze(1)

        s = hadamard(s, split_tensor[1])
        out_s = torch.add(s, split_tensor[2])

        return out_v, out_s


def positional_adjacency(molecule_pos: list, r: int) -> list:
    adj_list = [[], []]

    for i, cur_atom in enumerate(molecule_pos):
        for j, adj_atom in enumerate(molecule_pos):
            if not i == j and np.linalg.norm(cur_atom - adj_atom) <= r:
                adj_list[0].append(i)
                adj_list[1].append(j)

    return adj_list


def r_ij_calc(adj_list, positions):
    r_ij = []
    for idx, i in enumerate(adj_list[0]):
        r_ij.append(positions[adj_list[0][idx]] - positions[adj_list[1][idx]])
    return np.array(r_ij)


# def sum_over_j(v,idx_j):


if __name__ == "__main__":
    print("hej")

    numbers = np.array([6, 1, 1, 1])
    positions = np.array(
        [
            [-0.0126981359, 1.0858041578, 0.0080009958],
            [0.002150416, -0.0060313176, 0.0019761204],
            [1.0117308433, 1.4637511618, 0.0002765748],
            [-0.540815069, 1.4475266138, -0.8766437152],
        ]
    )

    n = 12

    # atomic_number = torch.randint(low=0, high=8, size=(n, 1))
    # pos = torch.randn(n, 3)
    model = painn()

    print(model(numbers, positions))
