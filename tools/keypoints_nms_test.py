import torch
import numpy as np
import os
from time import time

#Input: dets Tensor[N, numkeypoints*3+score]
def _oks_nms(dets, thresh=0.2):
    sigmas = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89], device=dets.device)/10.0
    vars = (sigmas * 2)**2
    vars = vars.unsqueeze(0).unsqueeze(0) #[1, 1, 17]

    #get pointsets from dets, which are [N, num_points*2], where each pointset is [x1, y1, x2, y2...]
    pointsets = dets[:,:-1]
    pointsets = pointsets.view((pointsets.size()[0], -1, 3))
    pointsets = pointsets[:,:,:2]

    w_all = torch.max(pointsets[:,:,0], dim=1)[0] - torch.min(pointsets[:,:,0], dim=1)[0]
    h_all = torch.max(pointsets[:,:,1], dim=1)[0] - torch.min(pointsets[:,:,1], dim=1)[0]
    areas = w_all*h_all
    areas = areas.clamp(32*32)
    areas = (areas.unsqueeze(0)+areas.unsqueeze(1))/2
    areas = areas.unsqueeze(-1) #[points_num, points_num, 1]

    distance = ((pointsets.unsqueeze(0) - pointsets.unsqueeze(1))**2).sum(dim=-1) # [m, m, points_num]
    oks = torch.exp(-distance/vars/areas).mean(dim=-1)

    scores = dets[:,-1]

    keep = []
    index = scores.sort(descending=True)[1]  

    while index.size()[0] >0:
        i = index[0]       # every time the first is the biggst, and add it directly
        keep.append(i)
        if index.size()[0] == 1:
            break       
        oks_selected = torch.index_select(oks[i], 0, index)
        idx = torch.where(oks_selected<=thresh)[0]
         
        index = index[idx]

    keep = torch.stack(keep)    
    return keep

def load_results(file_path, device='gpu'):
    results = np.load(file_path)
    if device=='gpu':
        results = torch.from_numpy(results).cuda()
    elif device=='cpu':
        results = torch.from_numpy(results)
    return results

def main():
    #Tow prepredicted-result sets
    #1. ./debug_img' (with score>0.05)
    #2. ./debug_img2' (with score>0.01)
    results_path = './debug_img'
    # results_path = './debug_img2'

    device = 'gpu' # device to run nms, 'gpu' or 'cpu'
    # device = 'cpu' 

    file_names = os.listdir(results_path)
    #record information
    total_candidates = 0
    warm_up = 100
    time_count = 0
    num_count = 0
    for i, file_name in enumerate(file_names):
        #load results
        file_path = os.path.join(results_path, file_name)
        results = load_results(file_path, device=device)
        num_cadidates = results.size()[0]

        torch.cuda.synchronize()
        t1 = time()

        keep = _oks_nms(results) #run keypoints nms

        torch.cuda.synchronize()
        t2 = time()    

        if i > warm_up:
            time_count += t2 - t1
            num_count += 1
            total_candidates += num_cadidates

    print('Average inference time (ms) %.2f'%(time_count/num_count*1000))
    print('Total images', num_count)
    print('Total candidates', total_candidates)
    print('Average candidates per image', total_candidates/num_count)


if __name__ == '__main__':
    main()