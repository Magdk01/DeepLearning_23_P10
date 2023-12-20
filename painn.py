import torch
import torch.nn as nn
import copy
import numpy as np


def hadamard(m1, m2):
    # assert m1.shape == m2.shape
    return m1 * m2


class painn(nn.Module):
    def __init__(self, r_cut=2, n=20, f=128, shared=True, device="cpu") -> None:
        super(painn, self).__init__()
        self.device = device
        self.r_cut = r_cut
        self.n = n
        self.f = f
        self.embedding_layer = nn.Sequential(nn.Embedding(10, self.f))
        # Message layers

        if not shared:
            self.ø_layer_1 = nn.Sequential(
                nn.Linear(self.f, self.f), nn.SiLU(), nn.Linear(self.f, 3 * self.f)
            )
            self.w_layer_1 = nn.Sequential(nn.Linear(20, 3 * self.f, bias=True))
            self.message_model_1 = message(
                self.r_cut,
                self.n,
                self.ø_layer_1,
                self.w_layer_1,
                self.f,
                device=self.device,
            )

            self.ø_layer_2 = nn.Sequential(
                nn.Linear(self.f, self.f), nn.SiLU(), nn.Linear(self.f, 3 * self.f)
            )
            self.w_layer_2 = nn.Sequential(nn.Linear(20, 3 * self.f, bias=True))
            self.message_model_2 = message(
                self.r_cut,
                self.n,
                self.ø_layer_2,
                self.w_layer_2,
                self.f,
                device=self.device,
            )

            self.ø_layer_3 = nn.Sequential(
                nn.Linear(self.f, self.f), nn.SiLU(), nn.Linear(self.f, 3 * self.f)
            )
            self.w_layer_3 = nn.Sequential(nn.Linear(20, 3 * self.f, bias=True))
            self.message_model_3 = message(
                self.r_cut,
                self.n,
                self.ø_layer_3,
                self.w_layer_3,
                self.f,
                device=self.device,
            )

            self.message_models = [
                self.message_model_1,
                self.message_model_2,
                self.message_model_3,
            ]

        else:
            self.shared_ø_layer = nn.Sequential(
                nn.Linear(self.f, self.f), nn.SiLU(), nn.Linear(self.f, 3 * self.f)
            )
            self.w_layer = nn.Sequential(nn.Linear(20, 3 * self.f, bias=True))

            message_model = message(
                self.r_cut,
                self.n,
                self.shared_ø_layer,
                self.w_layer,
                self.f,
                device=self.device,
            )
            self.message_models = [message_model] * 3

        # Update layers

        if not shared:
            self.a_1 = nn.Sequential(
                nn.Linear(2 * self.f, self.f), nn.SiLU(), nn.Linear(self.f, 3 * self.f)
            )
            self.V_1 = nn.Sequential(nn.Linear(self.f, self.f, bias=False))
            self.U_1 = nn.Sequential(nn.Linear(self.f, self.f, bias=False))

            self.update_model_1 = update(
                self.a_1, self.V_1, self.U_1, self.f, device=self.device
            )

            self.a_2 = nn.Sequential(
                nn.Linear(2 * self.f, self.f), nn.SiLU(), nn.Linear(self.f, 3 * self.f)
            )
            self.V_2 = nn.Sequential(nn.Linear(self.f, self.f, bias=False))
            self.U_2 = nn.Sequential(nn.Linear(self.f, self.f, bias=False))

            self.update_model_2 = update(
                self.a_2, self.V_2, self.U_2, self.f, device=self.device
            )

            self.a_3 = nn.Sequential(
                nn.Linear(2 * self.f, self.f), nn.SiLU(), nn.Linear(self.f, 3 * self.f)
            )
            self.V_3 = nn.Sequential(nn.Linear(self.f, self.f, bias=False))
            self.U_3 = nn.Sequential(nn.Linear(self.f, self.f, bias=False))

            self.update_model_3 = update(
                self.a_3, self.V_3, self.U_3, self.f, device=self.device
            )

            self.update_models = [
                self.update_model_1,
                self.update_model_2,
                self.update_model_3,
            ]
        else:
            self.shared_a = nn.Sequential(
                nn.Linear(2 * self.f, self.f), nn.SiLU(), nn.Linear(self.f, 3 * self.f)
            )

            self.shared_V = nn.Sequential(nn.Linear(self.f, self.f, bias=False))
            self.shared_U = nn.Sequential(nn.Linear(self.f, self.f, bias=False))

            update_model = update(
                self.shared_a, self.shared_V, self.shared_U, self.f, device=self.device
            )
            self.update_models = [update_model] * 3

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(self.f, self.f), nn.SiLU(), nn.Linear(self.f, self.f)
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
        self.v = torch.zeros((self.s.shape[0], 3, self.f))

        for idx in range(3):
            v, s = self.v.detach().clone().to(self.device), self.s.detach().clone().to(
                self.device
            )

            self.v, self.s = self.message_models[idx](
                self.s[self.idx_i], self.r, v[self.idx_i], self.idx_i
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
        sums = torch.zeros(num_items, dtype=torch.float32).to(self.device)
        indicies = torch.tensor(graph_indicies)
        # Add the data to the sums tensor at the specified indexes
        out = torch.squeeze(out)
        out = torch.sum(out, dim=-1).to(self.device)
        sums.index_add_(0, indicies, out)

        return sums


class message(nn.Module):
    def __init__(self, r_cut, n, ø_layer, w_layer, f=128, device="cpu") -> None:
        super(message, self).__init__()
        self.device = device
        self.ø = ø_layer.to(self.device)
        self.r_cut = r_cut
        self.n = n
        self.f = f
        # self.internal_w_layer = nn.Sequential(nn.Linear(20, 3 * self.f, bias=True))
        self.internal_w_layer = w_layer.to(self.device)

    def forward(self, s, r, v, idx_i):
        # s-block
        ø_out = self.ø(s)
        org_r = r.detach().clone()
        assert ø_out.size(2) == 3 * self.f

        # left r-block
        r = self.__rbf(r, self.n).to(self.device)
        r = self.__fcut(r, self.r_cut).to(self.device)
        w = self.internal_w_layer(r).to(self.device)
        # w = self.__fcut(r, self.r_cut)

        assert w.size(2) == 3 * self.f

        split = hadamard(w, ø_out).to(self.device)
        split_tensor = torch.split(split, self.f, dim=2)

        out_s = torch.zeros(len(set(idx_i)), 1, self.f).to(self.device)
        for idx, i in enumerate(idx_i):
            out_s[i] += split_tensor[1][idx]

        # right r-block
        org_r = org_r / torch.norm(org_r)
        org_r = org_r.unsqueeze(2).repeat(1, 1, self.f)
        org_r = hadamard(split_tensor[2], org_r)

        # v-block
        v = hadamard(split_tensor[0], v)
        v = torch.add(org_r, v)

        out_v = torch.zeros(len(set(idx_i)), 3, self.f).to(self.device)
        for idx, i in enumerate(idx_i):
            out_v[i] += v[idx]

        return out_v, out_s

    def __rbf(self, input, n):
        n_values = torch.arange(1, n + 1).float().to(self.device)  # Shape: (n,)
        r_norms = torch.norm(input, dim=1, keepdim=True).to(self.device)  # Shape: (12,)
        # Broadcasting r_norms to (12, n) and n_values to (12, n)
        r_norms = r_norms.unsqueeze(2).expand(-1, -1, n)
        n_values = n_values.unsqueeze(0).expand(r_norms.shape[0], 1, -1)
        return torch.sin((n_values * torch.pi) / self.r_cut * r_norms) / r_norms

    def __fcut(self, tensor, R_c):
        # Applying the cosine cutoff function element-wise
        cos_cutoff = torch.where(
            tensor <= R_c,
            0.5 * (torch.cos(torch.pi * tensor / R_c) + 1),
            torch.zeros_like(tensor),
        )
        return cos_cutoff


class update(nn.Module):
    def __init__(self, a, V, U, f=128, device="cpu") -> None:
        super(update, self).__init__()
        self.a = a
        self.V = V
        self.U = U
        self.f = f
        self.device = device

    def forward(self, s, v):
        # top v-block
        v = self.V(v)

        u = self.U(v)

        # s-block
        s_stack = torch.cat((torch.norm(v, dim=1, keepdim=True), s), axis=2)
        split = self.a(s_stack)
        split_tensor = torch.split(split, self.f, dim=2)
        # left v-block continues
        out_v = hadamard(u, split_tensor[0])

        # right v-block continues

        s = torch.sum((u * v), dim=1).unsqueeze(1)

        s = hadamard(s, split_tensor[1])
        out_s = torch.add(s, split_tensor[2])

        return out_v, out_s


def positional_adjacency(molecule_pos, r, graph_indices):
    adj_list = [[], []]

    # Use broadcasting to find pairwise distances
    dist_matrix = torch.norm(molecule_pos[:, None] - molecule_pos, dim=2)

    # Create adjacency matrix based on distance and graph index criteria
    adj_matrix = (dist_matrix <= r) & (graph_indices[:, None] == graph_indices)
    adj_matrix.fill_diagonal_(0)  # Remove self-loops

    # Convert adjacency matrix to edge list
    adj_list = adj_matrix.nonzero(as_tuple=False).t().tolist()

    return adj_list


def r_ij_calc(adj_list, positions):
    # Calculate r_ij for each pair
    r_ij = positions[adj_list[0]] - positions[adj_list[1]]

    return r_ij


if __name__ == "__main__":
    print("hej")
    from DataLoader.DataLoader import DataLoad
    from trainer import extract_and_calc_loss

    train_loader, test_loader, val_loader = DataLoad(batch_size=4, target_index=0)
    for batch, batch_indexies in train_loader:
        concatenated_list = [
            torch.cat(elements) if not idx == 2 else torch.tensor([elements])
            for idx, elements in enumerate(zip(*batch))
        ]
        loss = extract_and_calc_loss(
            concatenated_list, painn(), torch.nn.MSELoss(), batch_indexies
        )
        print(loss)

        break
