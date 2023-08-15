import numpy as np
import matplotlib.pyplot as plt

def create_and_save_heatmap(npy_path, image_path):
    # Load the data from the .npy file
    data = np.load(npy_path)[0,0,:,:]

    # Create the heatmap
    plt.figure(figsize=(10,10))
    plt.imshow(data, cmap='viridis') # 'viridis' is a green-blue color scheme

    # Remove the axes for a cleaner look
    plt.axis('off')

    # Save the heatmap to a .png file
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)

    # Close the plot to free up memory
    plt.close()

create_and_save_heatmap("/home/dogus/final_ws/solov2_venv/src/AdelaiDet/seg.npy", "/home/dogus/final_ws/solov2_venv/src/AdelaiDet/seg.png")
create_and_save_heatmap("/home/dogus/final_ws/solov2_venv/src/AdelaiDet/kernel.npy", "/home/dogus/final_ws/solov2_venv/src/AdelaiDet/kernel.png")