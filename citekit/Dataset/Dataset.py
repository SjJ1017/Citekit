from torch.utils.data import Dataset
import json

default_get = lambda key :  lambda data: data[key]

class PromptDataset(Dataset):

    def __init__(self,data_dir,*keys,**projections) -> None:
        self.data = []
        for d in data_dir:
            list_contents = {key:default_get(key)(d) for key in keys if key in d.keys()}
            dict_contents = {projection:projections[projection](d) for projection in projections.keys()}
            self.data.append({**list_contents,**dict_contents})
    
    def __getitem__(self, index) -> dict:
    
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

class FileDataset(PromptDataset):

    def __init__(self,data_dir,*keys,**projections) -> None:
        with open(data_dir,'r',encoding='utf-8') as file:
            data_dir = json.load(file)
        if not keys:
            keys = data_dir[0].keys()

        self.data = []
        for d in data_dir:
            list_contents = {key:default_get(key)(d) for key in keys if key in d.keys()}
            dict_contents = {projection:projections[projection](d) for projection in projections.keys()}
            self.data.append({**list_contents,**dict_contents})
    
    def __getitem__(self, index) -> dict:
    
        return self.data[index]
    
    def __len__(self):
        return len(self.data)