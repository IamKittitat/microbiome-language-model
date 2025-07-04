from torch.utils.data import Dataset,WeightedRandomSampler
import tqdm
import torch
import random
import pdb
import numpy as np

class ELECTRADataset(Dataset):
    def __init__(self, samples, embedding_path,labels,index=None):
        self.embeddings = np.load(embedding_path)
        self.samples = samples
        self.labels = labels
        self.seq_len = self.samples.shape[1]+1
        #Initialize cls token vector values
        #pdb.set_trace()

        #take average of all embeddings
        #self.cls = np.average(self.embeddings,axis=0)
        self.cls = np.zeros(self.embeddings.shape[1])
        self.frequency_index = self.samples.shape[2] - 1 
        self.cls_frequency = 1
        self.index=index


        #initialize mask token vector values

        #find max and min ranges of values for every feature in embedding space
        #create random embedding
        self.embedding_mins = np.amin(self.embeddings,axis=0)
        self.embedding_maxes = np.amin(self.embeddings,axis=0)
        self.mask = self.generate_random_embedding()


        self.padding = np.zeros(self.embeddings.shape[1])
        #add cls, mask, and padding embeddings to vocab embeddings
        self.embeddings = np.concatenate((self.embeddings,np.expand_dims(self.mask,axis=0)))
        self.embeddings = np.concatenate((self.embeddings,np.expand_dims(self.cls,axis=0)))
        self.embeddings = np.concatenate((self.embeddings,np.expand_dims(self.padding,axis=0)))
        
        self.mask_index = self.lookup_embedding(self.mask)
        self.cls_index = self.lookup_embedding(self.cls)
        self.padding_index = self.lookup_embedding(self.padding)
        

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, item):
        
        #pdb.set_trace()
        sample = self.samples[item]
        sorted_indices = np.argsort(sample[:,1])
        sample = sample[sorted_indices][::-1]
        cls_marker = np.array([[self.cls_index,self.cls_frequency]],dtype=np.float64)
        sample = np.concatenate((cls_marker,sample))
        electra_input,frequencies = self.match_sample_to_embedding(sample)
        electra_label = self.labels[item]       
        if self.index is not None:
            electra_label = np.asarray([-100 if i is not self.index else x for i,x in enumerate(electra_label)])
        output = {"electra_input": torch.tensor(electra_input,dtype=torch.long),
                "electra_label": torch.tensor(electra_label,dtype=torch.long),
                "species_frequencies": torch.tensor(frequencies,dtype=torch.long),
                }

        return output

    def match_sample_to_embedding(self, sample):
        electra_input = sample[:,0].copy()
        frequencies = np.zeros(sample.shape[0])
        for i in range(sample.shape[0]):
            #pdb.set_trace()
            if sample[i,self.frequency_index] > 0:
                frequencies[i] = sample[i,self.frequency_index]
            else:
                electra_input[i] = self.padding_index
                

        return electra_input,frequencies

    def generate_random_frequency(self):
        return np.random.randint(self.frequency_min,self.frequency_max)

    def generate_random_embedding(self):
        return np.random.uniform(self.embedding_mins,self.embedding_maxes)

    def vocab_len(self):
        return self.embeddings.shape[0]

    def lookup_embedding(self,bug):
        return np.where(np.all(self.embeddings == bug,axis=1))[0][0]

#for creating weighted random sampler
def create_weighted_sampler(labels,index):
    labels_unique, counts = np.unique(labels[:,index],return_counts=True)
    total_examples = 0
    for c,label in zip(counts,labels_unique):
        if label != -100:
            total_examples += c
    class_weights = {label:total_examples / c for c,label in zip(counts,labels_unique)}
    class_weights[-100] = 0
    #if index == 0:
    #    class_weights[1] = class_weights[1]/2
    example_weights = [class_weights[int(e)] for e in labels[:,index]]
    sampler = WeightedRandomSampler(example_weights,len(labels))
    return sampler
