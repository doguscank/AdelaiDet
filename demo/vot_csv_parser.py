import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Humans = ["agility", "basketball", "bolt1", "diver", "girl", "graduate", "gymnastics1", "gymnastics2", "gymnastics3", "handball1", "handball2", "iceskater1", "iceskater2", "marathon", "matrix", "polo", "rowing", "shaking", "singer2", "singer3", "soccer1", "soccer2", "soldier", "surfing"]
Vehicles = ["car1", "wiper"]
Animals = ["animal", "ants1", "birds1", "birds2", "butterfly", "crabs1", "fernando", "fish1", "fish2", "flamingo1", "kangaroo", "lamb", "monkey", "nature", "rabbit", "rabbit2", "snake", "tiger", "zebrafish"]
Others = ["bag", "ball2", "ball3", "book", "bubble", "conduction", "dinosaur", "drone1", "drone_across", "frisbee", "hand", "hand2", "helicopter", "leaves", "motocross1", "tennis", "wheel"]

def plot_weighted_averages(csv_files):
    for fn, l in zip(["humans", "vehicles", "animals", "others"], [Humans, Vehicles, Animals, Others]):
        # Create an empty dictionary to store the total_images and weighted averages
        weighted_averages = {"total_images": [], "find_rates": [], "acc": [], "miou": [], "dice": []}

        # Iterate over each csv file
        for file_name in csv_files:
            # Read the csv file into a pandas dataframe
            df = pd.read_csv(file_name)
            df = df[df["dataset"].isin(l)]

            # Calculate the weighted averages and add them to the dictionary
            total_images = df["total_images"].sum()
            weighted_averages["total_images"].append(total_images)
            weighted_averages["find_rates"].append((df["find_rates"] * df["total_images"]).sum() / total_images)
            weighted_averages["acc"].append((df["acc"] * df["total_images"]).sum() / total_images)
            weighted_averages["miou"].append((df["miou"] * df["total_images"]).sum() / total_images)
            weighted_averages["dice"].append((df["dice"] * df["total_images"]).sum() / total_images)

        # Create the four plots
        columns = ["find_rates", "acc", "miou", "dice"]
        names = ["IDR", "PA", "mIoU", "D"]

        # Create a single figure
        fig, ax = plt.subplots()

        # Iterate over each column and plot the corresponding weighted averages
        columns = ["find_rates", "acc", "miou", "dice"]
        colors = ["red", "blue", "green", "orange"]
        for i in range(4):
            x = np.arange(0.0, 1.0, 0.1)
            y = weighted_averages[columns[i]]
            ax.plot(x, y, color=colors[i], label=names[i], marker='o')

        # Add labels, title and legend
        ax.grid(True)
        ax.set_xlim(-0.025, 0.925)
        ax.set_xticks(np.arange(0., 1.0, 0.1))
        ax.set_yticks(np.arange(0., 1.1, 0.1))
        ax.set_ylim(-0.025, 1.025)
        ax.set_xlabel("Threshold value")
        ax.set_ylabel("Metric value")
        ax.set_title(f"Evaluation Metric Changes of {fn.capitalize()} Group")
        ax.legend()

        # plt.show()

        # for i in range(4):
        #     plt.subplot(2, 2, i+1)
        #     plt.plot(np.arange(0.0, 1.0, 0.1), weighted_averages[columns[i]], '-o')
        #     plt.title(names[i])
        #     plt.xlabel()
        #     plt.ylabel(f"{names[i]} value")

        # plt.suptitle(f"Evaluation Metric Changes of {fn.capitalize()} Group", fontsize=14)
        plt.tight_layout()

        # Save the plot as a png file
        plt.savefig(f"{fn}.png", dpi=300)
        plt.cla()
        plt.clf()

def generate_latex_table(dt, th):
    # Create separate dataframes for each list
    humans_df = dt[dt['dataset'].isin(Humans)]
    vehicles_df = dt[dt['dataset'].isin(Vehicles)]
    animals_df = dt[dt['dataset'].isin(Animals)]
    others_df = dt[dt['dataset'].isin(Others)]

    # Create the LaTeX table
    latex_table = "\\begin{longtable} {|c|c|c|c|c|c|c|c|c|}\n"
    latex_table += "\\caption{Visual Object Tracking 2022 metrics with a minimum IoU threshold value of " + str(th) + "}\n"
    latex_table += "\\label{tab:vot2022-metrics-" + str(th) + "} \\\\\n"
    latex_table += "\\hline Data Set & NI & NIF & IDR & PA & mIoU & $D$ & MRC & NIMRC \\\\\n"
    latex_table += "\\hline \\endfirsthead\n"
    latex_table += "\\multicolumn{9}{c}{{\\bfseries \\tablename\\ \\thetable{} -- continued from previous page}} \\\\\n"
    latex_table += "\\hline Data Set & NI & NIF & IDR & PA & mIoU & $D$ & MRC & NIMRC \\\\\n"
    latex_table += "\\hline \\endhead\n"
    latex_table += "\\hline \\multicolumn{9}{|r|}{{Continued on next page}} \\\\ \\hline \\endfoot\n"
    latex_table += "\\hline \\endlastfoot\n"

    # Add rows to the LaTeX table
    for df, name in zip([humans_df, vehicles_df, animals_df, others_df], ['Humans', 'Vehicles', 'Animals', 'Others']):
        latex_table += "\\multicolumn{9}{|l|}{{" + name + "}} \\\\ \\hline\n"
        ni_list = []
        nif_list = []
        idr_list = []
        pa_list = []
        miou_list = []
        d_list = []
        mrc_list = []
        nimrc_list = []
        for index, row in df.iterrows():
            # Extract the values from the row
            dataset = row['dataset']
            ni = row['total_images']
            nif = row['found']
            idr = row['find_rates']
            pa = row['acc']
            miou = row['miou']
            d = row['dice']

            # Get the most recurring class and its count
            mrc = row['most_repetitive_class']
            nimrc = row['most_repetitive_class_count']

            # Add the values to the corresponding list
            ni_list.append(ni)
            nif_list.append(nif)
            idr_list.append(idr)
            pa_list.append(pa)
            miou_list.append(miou)
            d_list.append(d)
            mrc_list.append(mrc)
            nimrc_list.append(nimrc)

            # Write the row to the LaTeX table
            latex_table += f"{dataset} & {ni} & {nif} & {idr:.3f} & {pa:.3f} & {miou:.3f} & {d:.3f} & {mrc} & {nimrc} \\\\ \n"

        ni_list = np.array(ni_list)
        nif_list = np.array(nif_list)
        idr_list = np.array(idr_list)
        pa_list = np.array(pa_list)
        miou_list = np.array(miou_list)
        d_list = np.array(d_list)
        mrc_list = np.array(mrc_list)
        nimrc_list = np.array(nimrc_list)

        # Write the average and weighted average rows to the LaTeX table
        avg_row_str = f"\\hline \n \\multicolumn{{3}}{{|l|}}{{Average values}} & {sum(idr_list)/len(idr_list):.3f} & {sum(pa_list)/len(pa_list):.3f} & {sum(miou_list)/len(miou_list):.3f} & {sum(d_list)/len(d_list):.3f} & \\multicolumn{{2}}{{|l|}}{{}} \\\\ \hline"
        latex_table += (avg_row_str + '\n')

        wavg_row_str = f"\\multicolumn{{3}}{{|l|}}{{Weighted average values}} & {sum(idr_list*ni_list)/sum(ni_list):.3f} & {sum(pa_list*ni_list)/sum(ni_list):.3f} & {sum(miou_list*ni_list)/sum(ni_list):.3f} & {sum(d_list*ni_list)/sum(ni_list):.3f} & \\multicolumn{{2}}{{|l|}}{{}} \\\\ \hline"
        latex_table += (wavg_row_str + '\n')

        # Write the closing statements for the LaTeX table
    latex_table += ("\\end{longtable}")

    return latex_table.replace(" nan ", " ").replace("_", r"\_")

csv = pd.read_csv("/home/dogus/Desktop/adelai/results_0_7.csv")

plot_weighted_averages([f"/home/dogus/Desktop/adelai/results_0_{str(x)}.csv" for x in range(10)])
exit()

print(generate_latex_table(
    csv, 0.7
))

