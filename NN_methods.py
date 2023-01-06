import numpy as np
from tqdm import tqdm
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data, InMemoryDataset
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool
import os.path as osp
import glob

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(2, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin   = Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 1. Obtain node embeddings 
        x = self.conv1(x.float(), edge_index)
        x = x.relu()
        x = self.conv2(x.float(), edge_index)
        x = x.relu()
        x = self.conv3(x.float(), edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        
        return x
    
    def loss(self, pred, score):
        # Start with MAE loss function, then switch to MSE when the error falls below delta
        return F.huber_loss(pred, score, reduction = 'mean', delta = 0.1) # ATTENTION delta dépend des labels 0.1 --> rmax =1 

class Simple_GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(Simple_GCN, self).__init__()
        self.conv1 = GCNConv(2, hidden_channels)
        self.lin   = Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 1. Obtain node embeddings 
        x = self.conv1(x.float(), edge_index)
        x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        
        return x
    
    def loss(self, pred, score):
        # Start with MAE loss function, then switch to MSE when the error falls below delta
        return F.huber_loss(pred, score, reduction = 'mean', delta = 0.1) # ATTENTION delta dépend des labels 0.1 --> rmax =1 

class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(2, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin   = Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 1. Obtain node embeddings 
        x = self.conv1(x.float(), edge_index)
        x = x.relu()
        x = self.conv2(x.float(), edge_index)
        x = x.relu()
        x = self.conv3(x.float(), edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        
        return x
    
    def loss(self, pred, score):
        # Start with MAE loss function, then switch to MSE when the error falls below delta
        return F.huber_loss(pred, score, reduction = 'mean', delta = 0.1) # ATTENTION delta dépend des labels 0.1 --> rmax =1 


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True



def save_checkpoint(epoch, loss, timestamp, model, optimizer):
    print('A checkpoint has been saved!')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, 'History/checkpoint' + timestamp + '.pt')

def load_checkpoint(model, optimizer, device, load_path):
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch

def save_history(epochs, optimizer_param, loss, timestamp):
    torch.save({
            'epoch': epochs,
            'optimizer_state_dict': optimizer_param,
            # we don't save model weights and biases every epochs to save memory 
            'Loss': loss,
            }, 'History/history' + timestamp + '.pt')



def create_graph(points, edge_index_, area):
    '''Create a Data object with points coordinates and volumes'''

    # Add coordinates as features
    sample = torch.tensor(points, dtype=torch.float32)
    # Add volume as label
    label = torch.tensor([area], dtype=torch.float32)
    # Create edge_index
    edge_index = torch.tensor(edge_index_, dtype = torch.long)
    # Create Data object
    graph = Data(x=sample, edge_index=edge_index, y=label)
    return graph


class EllipsesDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.samples_per_file = 50000
    @property
    def raw_file_names(self):
        return ['Three_Circles_cartesian0.npz',
                'Three_Circles_cartesian1.npz',
                'Three_Circles_polar3.npz',
                'Three_Circles_polar4.npz',
                'Truncated_Once_cartesian0.npz',
                'Truncated_Once_cartesian1.npz',
                'Truncated_Once_polar3.npz',
                'Truncated_Once_polar4.npz',
                'Truncated_Twice_cartesian0.npz',
                'Truncated_Twice_cartesian1.npz',
                'Truncated_Twice_polar3.npz',
                'Truncated_Twice_polar4.npz',
                'Two_Circles_cartesian0.npz',
                'Two_Circles_cartesian1.npz',
                'Two_Circles_polar3.npz',
                'Two_Circles_polar4.npz',
                'Unique_Ellipse_cartesian0.npz',
                'Unique_Ellipse_cartesian1.npz',
                'Unique_Ellipse_polar2.npz',
                'Unique_Ellipse_polar3.npz']

    @property
    def processed_file_names(self):
        return [f'data_{idx}.dt' for idx in range(18)]

    def process(self):
        for k in range(len(self.processed_file_names)):
            data_list = []
            for j in range(len(self.raw_file_names)):
                filename = self.raw_file_names[j]
                print('--- File ', j+1, '/', len(self.raw_file_names), ' ---')
                with np.load(self.root + '/raw/' + filename, allow_pickle = True) as data:
                    coords     = data['coords']
                    slices     = data['slices'].astype(int)
                    labels     = data['labels']
                    edge_index = data['edge_index']
                
                n_samples = round((len(slices) - 1)/20)
                print("There are ",n_samples, " samples.")
                for i in tqdm(range(k*n_samples, (k+1)*n_samples)):
                    idx1 = slices[i]
                    idx2 = slices[i+1]
                    data_list.append(create_graph(coords[idx1:idx2, :], edge_index[:, idx1*10:idx2*10], labels[i]))
                
                del coords
                del slices
                del labels
                del edge_index

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[k])
    
    def __len__(self):
        return int(len(self.processed_paths)*self.samples_per_file)

    def __getitem__(self, idx):
        # compute the file number rounding down the division below
        k = int(idx/self.samples_per_file)
        
        # load the file k
        data, dict = torch.load(self.processed_paths[k])

        # compute the index of the slice
        s_idx = idx%self.samples_per_file
        # Extract sample in file k at index s_idx
        x = data.x[dict['x'][s_idx].item():dict['x'][s_idx+1].item()] 
        y = data.y[dict['y'][s_idx].item():dict['y'][s_idx+1].item()]
        edge_index = data.edge_index[:, dict['edge_index'][s_idx].item():dict['edge_index'][s_idx+1].item()] 

        return Data(x=x, edge_index=edge_index, y=y)



class test(InMemoryDataset):
    def __init__(self, root, dataset_nb ,transform=None, pre_transform=None, pre_filter=None):
        self.data_nb = dataset_nb
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])



    @property
    def raw_file_names(self):
        return ['Three_Circles_cartesian0.npz',
                'Three_Circles_cartesian1.npz',
                'Three_Circles_polar3.npz',
                'Three_Circles_polar4.npz',
                'Truncated_Once_cartesian0.npz',
                'Truncated_Once_cartesian1.npz',
                'Truncated_Once_polar3.npz',
                'Truncated_Once_polar4.npz',
                'Truncated_Twice_cartesian0.npz',
                'Truncated_Twice_cartesian1.npz',
                'Truncated_Twice_polar3.npz',
                'Truncated_Twice_polar4.npz',
                'Two_Circles_cartesian0.npz',
                'Two_Circles_cartesian1.npz',
                'Two_Circles_polar3.npz',
                'Two_Circles_polar4.npz',
                'Unique_Ellipse_cartesian0.npz',
                'Unique_Ellipse_cartesian1.npz',
                'Unique_Ellipse_polar2.npz',
                'Unique_Ellipse_polar3.npz']

    @property
    def processed_file_names(self):
        return ['data_' + str(self.data_nb) + '.dt']

    def process(self):
        j = 0
        data_list = []
        for filename in self.raw_file_names:
            print('--- File ', j+1, '/', len(self.raw_file_names), ' ---')
            with np.load(osp.join(self.raw_dir, filename), allow_pickle = True) as data:
                coords     = data['coords']
                slices     = data['slices'].astype(int)
                labels     = data['labels']
                edge_index = data['edge_index']
                
            for i in tqdm(range(5000)):
                idx1 = slices[i]
                idx2 = slices[i+1]
                data_list.append(create_graph(coords[idx1:idx2, :], edge_index[:, idx1*10:idx2*10], labels[i]))
            j += 1
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        

class test2(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return ['Three_Circles_cartesian0.npz',
                'Three_Circles_cartesian1.npz',
                'Three_Circles_polar3.npz',
                'Three_Circles_polar4.npz',
                'Truncated_Once_cartesian0.npz',
                'Truncated_Once_cartesian1.npz',
                'Truncated_Once_polar3.npz',
                'Truncated_Once_polar4.npz',
                'Truncated_Twice_cartesian0.npz',
                'Truncated_Twice_cartesian1.npz',
                'Truncated_Twice_polar3.npz',
                'Truncated_Twice_polar4.npz',
                'Two_Circles_cartesian0.npz',
                'Two_Circles_cartesian1.npz',
                'Two_Circles_polar3.npz',
                'Two_Circles_polar4.npz',
                'Unique_Ellipse_cartesian0.npz',
                'Unique_Ellipse_cartesian1.npz',
                'Unique_Ellipse_polar2.npz',
                'Unique_Ellipse_polar3.npz']

    @property
    def processed_file_names(self):
        return ['data_0.dt']

    def process(self):
        j = 0
        data_list = []
        for filename in self.raw_file_names:
            print('--- File ', j+1, '/', len(self.raw_file_names), ' ---')
            with np.load(osp.join(self.raw_dir, filename), allow_pickle = True) as data:
                coords     = data['coords']
                slices     = data['slices'].astype(int)
                labels     = data['labels']
                edge_index = data['edge_index']
            
            #Min Max Normalization:
            labels = [label/(np.pi*10**2) for label in labels]
            
            for i in tqdm(range(round(len(labels)/4))):
                idx1 = slices[i]
                idx2 = slices[i+1]
                data_list.append(create_graph(coords[idx1:idx2, :], edge_index[:, idx1*10:idx2*10], labels[i]))
            j += 1
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class Circles_Dataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.files = glob.glob(root + '/*')
        super().__init__(root, transform, pre_transform, pre_filter)


    @property
    def processed_file_names(self):
        return self.files

    def len(self):
        return len(self.files)

    def get(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = np.load(self.files[idx], allow_pickle=True)
        edge_index = data['edge_index']
        
        # normalize the coordinates by the maximum length during the creation of samples  
        coords = data['x']/15 

        # Normalize the area by the maximum area possible : 15x15x4
        label = data['label']/(15*15*4)

        graph = create_graph(coords, edge_index, label)

        return graph