import numpy as np

def positional_adjacency(
    molecule_pos: list,
    r:int
) -> list:
    adj_list = [[], []]
    
    for i, cur_atom in enumerate(molecule_pos):
        for j, adj_atom in enumerate(molecule_pos):
            if not i==j and np.linalg.norm(cur_atom - adj_atom) <= r:
                adj_list[0].append(i); adj_list[1].append(j) 
    
    return adj_list
    

if __name__ == "__main__":
    from torch_geometric.datasets import QM9
    dataset = QM9(root='./QM9')
    
    print(positional_adjacency(dataset[2].pos, 2))
