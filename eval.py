import numpy as np
import os, sys, pathlib
import pandas as pd
from addict import Dict
import yaml
import coastal_mapping.model.metrics as m

if __name__== "__main__":
    conf = Dict(yaml.safe_load(open('./conf/eval.yaml')))
    preds_dir = pathlib.Path(conf.preds_dir) / conf.model_name
    processed_dir = pathlib.Path(conf.processed_dir)
    types = pd.read_csv("./types.csv")
    types.dropna(inplace=True)
    types = types[types["type"] == 1]
    coastlines = list(types.slice)
    coastlines = [x.replace("img", "mask") for x in coastlines]
    
    preds = [x for x in os.listdir(preds_dir) if "mask" in x]
    df = pd.DataFrame(columns=['Filename', 'IoU', 'Precision', 'Recall', 'Tp', 'Fp', 'Fn'])
    if conf.label == "water":
        invert = False
    elif conf.label == "land":
        invert = True
    else:
        raise ValueError("Undefined Label")

    for f in preds:
        if f.split(".")[0] in coastlines:
            true = np.squeeze(np.load(processed_dir / f)).astype(bool)
            preds = np.load(preds_dir / f) >= conf.threshold
            if invert:
                true = np.invert(true)
                preds = np.invert(preds)

            iou = m.IoU(preds, true)
            precision = m.precision(preds, true)
            recall = m.recall(preds, true)
            tp, fp, fn = m.tp_fp_fn(preds, true)
            new_row = {'Filename':f, 
                        'IoU':iou, 
                        'Precision':precision, 
                        'Recall':recall,
                        'Tp': tp,
                        'Fp': fp,
                        'Fn': fn
                    }
            #append row to the dataframe
            df = df.append(new_row, ignore_index=True)

    new_row = {'Filename':"Mean", 
                'IoU':np.round(np.sum(np.asarray(df.Tp))/(np.sum(np.asarray(df.Tp)) + np.sum(np.asarray(df.Fp))+ np.sum(np.asarray(df.Fn))), 4), 
                'Precision':np.round(np.sum(np.asarray(df.Tp))/(np.sum(np.asarray(df.Tp)) + np.sum(np.asarray(df.Fp))), 4), 
                'Recall':np.round(np.sum(np.asarray(df.Tp))/(np.sum(np.asarray(df.Tp)) + np.sum(np.asarray(df.Fn))), 4), 
                'Tp':np.round(np.sum(np.asarray(df.Tp)), 4), 
                'Fp':np.round(np.sum(np.asarray(df.Fp)), 4),
                'Fn':np.round(np.sum(np.asarray(df.Fn)), 4),
                }
    df = df.append(new_row, ignore_index=True)
    
    csv_name = "eval_"+conf.model_name+"_"+conf.label+".csv"
    df.to_csv(csv_name, sep=',', encoding='utf-8', index=False)