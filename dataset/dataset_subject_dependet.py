import torch 
import os 
import pickle as pik
import torch.nn as nn 
import random

def extract_and_tensor (path , dtype) : 
    if os.path.exists(path) : 
        with open(path , 'rb' ) as f : 
            data = pik.load(f)
        torch_data = torch.tensor(data , dtype = dtype  )
        return torch_data 

def slice_data(data, overlap, time_len):
    sampling_rate = 128
    slice_len = int(time_len * sampling_rate)
    #step = int(slice_len * (1 - overlap))
    overlap_samples = int(slice_len * (overlap / 100))
    step = slice_len - overlap_samples
    sliced_data = []
    for start_index in range(0, data.shape[0] - slice_len + 1, step):
        data_slice = data[start_index : start_index + slice_len, :]
        sliced_data.append(data_slice)
    if sliced_data:
        return torch.stack(sliced_data)
    else:
        return torch.empty(0, slice_len, data.shape[1], dtype=data.dtype)

class dataset(nn.Module) : 
    def __init__(self  , overlap  , time_lenght    , emotion , label_method  ) :
        super().__init__()
        self.overlap = overlap
        self.time_lenght = time_lenght
        self.emotion = emotion
        self.emotion_number = 0
        if self.emotion == 'valence' : 
            self.emotion_number  =1
        elif self.emotion=="dominance" : 
            self.emotion_number = 2
        self.label_method = label_method
        self.variance =  1
        self.mean = 0

    def extract (self , file_path , dtype, person , shuffle = False   ) : 
        base_extracted_dir = file_path['base_extracted_dir']
        label_file_path = file_path['labels_file']
        stimuli_files = file_path['stimuli_files']
        baseline_data = extract_and_tensor(base_extracted_dir, dtype) #(23, 18 , 7808, 14)
        labels = extract_and_tensor(label_file_path, dtype)

        all_train_data_slices = []
        all_train_label_slices = []

        for i in range(18):
            stimuli_data = extract_and_tensor(stimuli_files[i], dtype=dtype)
            person_stimuli = stimuli_data[person, :, :]
            sliced_stimuli = slice_data(person_stimuli, self.overlap, self.time_lenght)
            sliced_baseline= slice_data(baseline_data[person][i] , 0 , self.time_lenght)
            sliced_baseline = torch.sum(sliced_baseline , dim=0)/sliced_baseline.shape[0]
            sliced_stimuli -= sliced_baseline
            if sliced_stimuli is not None and sliced_stimuli.shape[0] > 0:
                all_train_data_slices.append(sliced_stimuli)
                
                num_slices = sliced_stimuli.shape[0]
                current_label = labels[person][i][self.emotion_number]
                sliced_labels = torch.full((num_slices,), current_label, dtype=torch.long)
                all_train_label_slices.append(sliced_labels)

        all_data = torch.cat(all_train_data_slices, dim=0) # (batch , time_len , 14)
        all_labels = torch.cat(all_train_label_slices, dim=0) #(batch,)
        if self.label_method == 'binary' : 
            all_labels = (all_labels > 2).long() 
        else : 
            all_labels -= 1 
            all_labels= all_labels.long()
        idx = list(range(all_labels.shape[0]))
        if shuffle : 
            random.shuffle(idx)
            all_data = all_data[idx , : , :]
            all_labels= all_labels[idx]
                # ✅ چک کردن لیبل‌ها
        print("Unique labels in dataset:", torch.unique(all_labels))
        print("Min label:", all_labels.min().item(), "Max label:", all_labels.max().item())
        print("Number of samples:", all_labels.shape[0])
        mean = all_data.mean()
        var  = all_data.var()
        if var ==0: 
            var = torch.tensor(1e-4 )
        std = torch.sqrt(var)
        all_data = (all_data - mean)/(std)
        return all_data , all_labels





