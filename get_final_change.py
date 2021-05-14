from PIL import Image
import os
import numpy as np 

inp_dir = "./changes/"
out_dir = "./final/"

files = sorted(os.listdir(inp_dir))

for i in range(1, len(files)):
    cur_fname = files[i]
    cur_zone = cur_fname.split("_")[0]
    cur_cube_id = "_".join(cur_fname.split("_")[1:4])
    cur_date = "-".join(cur_fname.split("_")[4].split(".")[0].split("-")[0:2])
    
    prev_fname = files[i-1]
    prev_zone = prev_fname.split("_")[0]
    prev_cube_id = "_".join(prev_fname.split("_")[1:4])
    prev_date = "-".join(prev_fname.split("_")[4].split(".")[0].split("-")[0:2])

    if cur_zone + "/" + cur_cube_id == prev_zone + "/" + prev_cube_id:
        filename = cur_zone + "_" + cur_cube_id + "-" + cur_date + "-" + prev_date + ".png"
        
        cur_img = np.load(inp_dir + cur_fname)
        prev_img = np.load(inp_dir + prev_fname)

        cur_img[cur_img == prev_img] = 0
        cur_img = cur_img.astype(np.uint8)
        
        image = Image.fromarray(cur_img)
        
        image.save(out_dir + filename)