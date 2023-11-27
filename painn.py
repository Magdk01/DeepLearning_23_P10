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

        # Message layers
        self.shared_ø_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 384),
        )
        message_model = message(self.r_cut, self.n, self.shared_ø_layer)
        self.message_models = [message_model] * 3

        # Update layers
        self.shared_a = nn.Sequential(
            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 384)
        )
        self.shared_V = nn.Sequential(nn.Linear(128, 128, bias=True))
        self.shared_U = nn.Sequential(nn.Linear(128, 128, bias=True))
        update_model = update(self.shared_a, self.shared_V, self.shared_U)
        self.update_models = [update_model] * 3

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128)
        )

    def forward(self, atomic_numbers, positional_encodings, graph_indicies):
        # Init for batch
        self.positions = torch.tensor(positional_encodings)
        self.graph_indicies = graph_indicies
        adj_list = positional_adjacency(self.positions, self.r_cut, self.graph_indicies)
        self.idx_i = adj_list[0]
        self.idx_j = adj_list[1]
        self.atomic = torch.tensor(atomic_numbers)
        self.r = torch.tensor(
            r_ij_calc(adj_list, positional_encodings), dtype=torch.float32
        )

        # Embedding of atoms
        self.s = self.embedding_layer(self.atomic).unsqueeze(1)

        # Then scaling to match all connections in molecule
        self.v = torch.zeros((self.s.shape[0], 3, 128))

        for idx in range(3):
            v, s = self.v.detach().clone(), self.s.detach().clone()

            self.v, self.s = self.message_models[idx](
                self.s[self.idx_j], self.r, v[self.idx_j], self.idx_i
            )

            self.v = self.v + v
            self.s = self.s + s

            v, s = self.v.detach().clone(), self.s.detach().clone()

            self.v, self.s = self.update_models[idx](s, v)

            self.v = self.v + v
            self.s = self.s + s

        out = self.output_layers(self.s)
        # Create a tensor for sums
        num_items = graph_indicies[-1] + 1  # Assuming indexes start from 0
        sums = torch.zeros(num_items, dtype=torch.float32)
        indicies = torch.tensor(graph_indicies)
        # Add the data to the sums tensor at the specified indexes
        out = torch.squeeze(out)
        out = torch.sum(out,dim=-1)
        sums.index_add_(0, indicies, out)


        return sums


class message(nn.Module):
    def __init__(self, r_cut, n, ø_layer) -> None:
        super(message, self).__init__()
        self.ø = ø_layer
        self.r_cut = r_cut
        self.n = n
        self.internal_w_layer = nn.Sequential(nn.Linear(20, 384, bias=True))

    def forward(self, s, r, v, idx_i):
        # s-block
        ø_out = self.ø(s)
        org_r = r.detach().clone()
        assert ø_out.size(2) == 384

        # left r-block
        r = self.__rbf(r, self.n)
        r = self.__fcut(r, self.r_cut)
        w = self.internal_w_layer(r)
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
    def __init__(self, a, V, U) -> None:
        super(update, self).__init__()
        self.a = a
        self.V = V
        self.U = U

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


def positional_adjacency(molecule_pos: list, r: int, graph_indicies: list) -> list:
    adj_list = [[], []]

    for i, cur_atom in enumerate(molecule_pos):
        for j, adj_atom in enumerate(molecule_pos):
            if (
                not i == j
                and np.linalg.norm(cur_atom - adj_atom) <= r
                and graph_indicies[i] == graph_indicies[j]
            ):
                adj_list[0].append(i)
                adj_list[1].append(j)

    return adj_list


def r_ij_calc(adj_list, positions):
    r_ij = []
    for idx, i in enumerate(adj_list[0]):
        pos = positions[adj_list[0][idx]] - positions[adj_list[1][idx]]
        r_ij.append(pos)
    return np.array([r_ij[0]])


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
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    print(model(numbers, positions))
