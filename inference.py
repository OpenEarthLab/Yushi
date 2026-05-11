import numpy as np
import os
import torch
import torch.nn.functional as F
from yushi_3d import YuShi3D
import matplotlib.pyplot as plt
import pdb
import pickle
import cmaps


device = torch.device("cuda")

def main():
    # Load data
    example_path = './input_examples/case_1.pkl' 
    with open(example_path, 'rb') as fd:
        data = pickle.load(fd)
    inp = data['inp']
    inp = torch.from_numpy(inp)
    inp = inp.cuda() #Input shape: [B, T, L, H, W], like [1, 10, 7, 461, 461]

    target = data['tar']

    # Input and output length
    input_len = 10
    pred_len = 20 


   # Initialize model
    model = YuShi3D() 
    checkpoint_path = os.path.join(os.getcwd(), 'model.ckpt')
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    # Forward pass
    with torch.no_grad():
        pred = model(inp)
    
    # Normalization
    pred = pred.detach().cpu().numpy() #Output shape: [B, T, L, H, W], like [1, 20, 7, 461, 461]
    pred = pred.max(axis=2)
    pred = np.clip(pred * 70.0, a_min=0., a_max=70.0)

    target = target.max(axis=2)
    target = np.clip(target * 70.0, a_min=0., a_max=70.0)

    # Visualization 
    fg, axes = plt.subplots(2, 4, figsize=(float(4 * 4.61), float(2 * 4.61)))

    for index, num in enumerate([4, 9, 14, 19]):
        axes[0, index].imshow(target[0, num][::-1, ...], cmap=cmaps.precip3_16lev, vmin=1.0, vmax=50.0)
        axes[1, index].imshow(pred[0, num][::-1, ...], cmap=cmaps.precip3_16lev, vmin=1.0, vmax=50.0)
    
    axes[0,0].text(-80, 250, 'Observation', va='center', ha='center', rotation=90, fontsize=25)
    axes[1,0].text(-80, 250, 'Yushi', va='center', ha='center', rotation=90, fontsize=25)

    axes[0,0].text(250, -50, 'T+30min', va='center', ha='center', fontsize=25)
    axes[0,1].text(250, -50, 'T+60min', va='center', ha='center', fontsize=25)
    axes[0,2].text(250, -50, 'T+90min', va='center', ha='center', fontsize=25)
    axes[0,3].text(250, -50, 'T+120min', va='center', ha='center', fontsize=25)


    for  row in range(2):
        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    
    plt.subplots_adjust(
        left=0.1,   
        right=0.9,  
        top=0.9,    
        bottom=0.1, 
        wspace=0.08, 
        hspace=0.08  
    )

    save_path = os.path.basename(example_path).replace('.pkl', '.png')
    save_path = os.path.join('./output_figures', save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()




    


    
if __name__ == '__main__':
    main()