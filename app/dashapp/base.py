from pathlib import Path
import pandas as pd
import os

class Base():
    def __init__(self):
        self.data_dir = Path("/data/baryal/HKH/")
        self.processed_folder = "processed_L07_2005"
        self.processed_dir = self.data_dir / self.processed_folder
        self.preds_folder = "1_rgb"
        self.preds_dir = self.data_dir / self.processed_folder / "preds" / self.preds_folder

    def get_df(self):
        return self.df

    def set_df(self):
        df = pd.read_csv(self.preds_dir / "metadata.csv", index_col=0)
        df = df.sort_values('tile_name').reset_index()
        df = df.round(4)
        df = df.drop(["index"], axis=1)
        df["id"] = df.index.tolist()
        self.df = df

    def set_preds_folder(self, foldername):
        self.preds_folder = foldername
        self.preds_dir = self.data_dir / self.processed_folder / "preds" / self.preds_folder
        self.set_df()

    def set_processed_folder(self, foldername):
        self.processed_folder = foldername
        self.processed_dir = self.data_dir / self.processed_folder

    def get_processed_folder(self):
        return self.processed_folder

    def get_data_dir(self):
        return self.data_dir

    def get_preds_dir(self):
        return self.preds_dir

    def get_preds_folder(self):
        return self.preds_folder

    def get_all_processed_folders(self):
        self.all_processed_folders = sorted([x for x in os.listdir(self.data_dir) if "processed" in x])
        return self.all_processed_folders

    def get_processed_dir(self):
        return self.processed_dir

    def get_all_preds_folders(self):
        self.all_preds_folders = sorted([x for x in os.listdir(self.processed_dir / "preds")])
        return self.all_preds_folders