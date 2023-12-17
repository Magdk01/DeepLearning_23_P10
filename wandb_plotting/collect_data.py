import wandb
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Initialize API
api = wandb.Api()

def get_data(entity, project_name, run_id, keys, samples = 100_000):
    if type(keys) != list:
        keys = [keys]
        
    run = api.run(f"{entity}/{project_name}/{run_id}")
    data = run.history(pandas = True, samples=samples, keys=keys)
    return data


def make_plot(data, title, y_title, filename=None, plot=False):  
    cols = data.columns[1:]
    # steps = data['_step']
    plt.figure(figsize=(10,5))
    plt.xticks(rotation=90)
    for col in cols:
        plt.barh(col.split(" // ")[1], data[col], label = col.split(" // ")[1])
    # plt.axhline(0.06, xmin=0.05, xmax=0.95, c= "r")
    plt.title(title.replace("_", " ").upper())
    plt.xlabel(y_title)
    # plt.ylabel(y_title)
    plt.grid()  
    plt.tight_layout()
    # plt.legend()
    
    if plot == "jpg":
        plt.savefig(f"./wandb_plotting/figures/{filename}.jpg")
    plt.show()
    

if __name__ == "__main__":
    pids = ["deep_learing_p10/painn/q9vckkk6", "deep_learing_p10/painn/st2llfzw", "deep_learing_p10/painn/qd9d8g9l", "deep_learing_p10/painn/fbnd75fj", "deep_learing_p10/painn/0rbwwvhr", "deep_learing_p10/painn/lqlbodio", "deep_learing_p10/painn/q3jlqktd", "deep_learing_p10/painn/39ufjbjy"]
    names = ["Heat capacity", "Zero point vibrational energy", "Electronic Spatial Extend", "Gap Between HOMO and LUMO", "Lowest unoccupied molecular orbital energy", "Highest occupied molecular orbital energy", "Isotropic polarizability", "Dipole Moment"]
    graphs = ["Test Loss"]#, "Mean Validation Loss"]
    y_titles = ["Mean Absolute Error"]#, "Mean Squared Error"]

    for j, g in enumerate(graphs):
        for i, pid in enumerate(pids):
            entity, project_name, run_id = pid.split("/")
            data = get_data(entity=entity, project_name=project_name, run_id=run_id, keys=g)
            data.columns = ["_step", g + " // " + names[i]]
            if i == 0:
                all_data = data
            else:
                all_data = pd.merge(all_data, data, on="_step", how="outer")
        
        make_plot(all_data, title = g, y_title = y_titles[j], filename=g.replace("/", "").replace(" ", "").lower() + "_painn", plot="jpg")