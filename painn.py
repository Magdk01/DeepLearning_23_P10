import torch
import torch.nn as nn
import copy


def hadamard(m1, m2):
    assert m1.shape == m2.shape
    return m1 * m2


class painn(nn.Module):
    def __init__(self, atomic_numbers, positional_encodings, r_cut=2, n=20) -> None:
        super(painn, self).__init__()
        self.atomic = atomic_numbers
        self.r = positional_encodings
        # self.v = torch.zeros_like(self.atomic)
        self.r_cut = r_cut
        self.n = n

        self.embedding_layer = nn.Sequential(nn.Embedding(9, 128))
        self.message_model = message(self.r_cut, self.n)
        self.update_model = update()

        self.output_layers = nn.Sequential(
            nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128)
        )

    def forward(self):
        self.s = self.embedding_layer(self.atomic)
        self.v = torch.zeros((self.r.shape[0], 3, 128))
        self.v = torch.broadcast_to(self.v, (self.v.shape[0], self.v.shape[1], 128))

        for _ in range(3):
            v, s = self.v.detach().clone(), self.s.detach().clone()

            self.v, self.s = self.message_model(s, self.r, v)

            self.v = torch.add(self.v, v)
            self.s = torch.add(self.s, s)

            v, s = self.v.detach().clone(), self.s.detach().clone()

            self.v, self.s = self.update_model(s, v)

            self.v = torch.add(self.v, v)
            self.s = torch.add(self.s, s)

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

    def forward(self, s, r, v):
        # s-block
        ø_out = self.ø(s)
        org_r = r.detach().clone()
        assert ø_out.size(2) == 384

        # left r-block
        r = self.__rbf(r, self.n)
        r = nn.Linear(20, 384)(r)
        w = self.__fcut(r, self.r_cut)

        assert w.size(2) == 384

        split = hadamard(w, ø_out)
        split_tensor = torch.split(split, 128, dim=2)

        out_s = torch.sum(split_tensor[1], axis=0)

        # right r-block
        org_r = org_r / torch.norm(org_r)
        org_r = torch.norm(org_r, dim=1).unsqueeze(1).unsqueeze(2).repeat(1, 1, 128)
        org_r = hadamard(split_tensor[2], org_r)  # To fix di

        # v-block
        v = hadamard(split_tensor[0], v)
        v = torch.add(org_r, v)

        out_v = torch.sum(v, axis=0)

        return out_v, out_s

    def __rbf(self, input, n):
        n_values = torch.arange(1, n + 1).float()  # Shape: (n,)
        r_norms = torch.norm(input, dim=1, keepdim=True)  # Shape: (12,)

        # Broadcasting r_norms to (12, n) and n_values to (12, n)
        r_norms = r_norms.unsqueeze(2).expand(-1, -1, n)

        n_values = n_values.unsqueeze(0).expand(12, 1, -1)

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
        s_stack = torch.stack((torch.norm(v), s))
        split = self.a(s_stack)

        # left v-block continues
        out_v = hadamard(u, split[:128])

        # right v-block continues
        s = torch.tensordot(v, u, dims=len(v))
        s = hadamard(s, split[128 : 128 * 2])
        out_s = torch.add(s, split[: 128 * 2])

        return out_v, out_s


if __name__ == "__main__":
    print("hej")
    n = 12
    atomic_number = torch.randint(low=0, high=8, size=(n, 1))
    pos = torch.randn(n, 3)
    model = painn(atomic_number, pos)

    print(model())
