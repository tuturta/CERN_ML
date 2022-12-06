# -*- coding: utf-8 -*-
# Create a GNN dataset with k-NNGraphs with Torch Geometric
from sample_methods import *
from NN_methods import *
import time
import torch
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter


'''----------------PARAMETERS-----------------'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps = 1e-7)
reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, factor=0.5, patience=4, min_lr=1e-7, verbose = True)
early_stopping = EarlyStopping(patience=10)
criterion = model.loss
writer = SummaryWriter()
timestamp = time.strftime("%Y%m%d-%H%M%S") 


batch_size= 64
epochs = 15


'''------------------FUNCTIONS-----------------'''

def fit(train_loader):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_mae = 0.
    train_running_mse = 0.
    counter = 0
    prog_bar = tqdm(enumerate(train_loader), total = len(train_loader))
    for data in prog_bar:  # Iterate in batches over the training dataset.
        counter += 1
        data = data[1].to(device)
        target = data.y.float()/(np.pi*10**2) # Normalize labels to 1
        # zero the parameter gradients
        model.zero_grad()
        preds = model(data)[:,0]  # Perform a single forward pass.
        loss = criterion(preds, target)
        # update metrics 
        train_running_loss += loss.item()
        train_running_mse += F.mse_loss(preds,target)
        train_running_mae += F.l1_loss(preds, target)

        loss.backward()
        # Update parameters based on gradients.
        optimizer.step() 

    # compute the mean metrics over the all batch
    train_loss = train_running_loss / counter
    train_mse = train_running_mse/counter
    train_mae = train_running_mae/counter

    return train_loss, train_mse, train_mae


def validate(val_dataloader):
    print('Validating')
    model.eval()
    val_running_loss = 0.
    val_running_mae = 0.
    val_running_mse = 0.
    counter = 0
    prog_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    with torch.no_grad():
        for i, data in prog_bar:
            counter += 1
            data = data.to(device)
            target = data.y.float()/(np.pi*10**2) # Normalize labels to 1
            preds = model(data)[:,0]
            loss = criterion(preds, target)
            # update metrics
            val_running_loss += loss.item()
            val_running_mse += F.mse_loss(preds,target)
            val_running_mae += F.l1_loss(preds, target)
    
    # Compute the mean of metrics over the all batch
    val_loss = val_running_loss / counter
    val_mse = val_running_mse/counter
    val_mae = val_running_mae/counter

    return val_loss, val_mse, val_mae


'''--------------MAIN------------'''

# Load the full dataset
dataset   = test(root='Data')
n_samples = len(dataset)

# Split into train and validation set
dataset = dataset.shuffle()
train_size = round(0.8*len(dataset))
val_size  = n_samples - train_size
train_dataset = dataset[:train_size]
val_dataset = dataset[val_size:]

# Create data loader for train and val sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = True) 


# Training loop

start = time.time()
loss = {'Hubert_train':[], 'Hubert_val': [],
        'MAE_train':   [], 'MAE_val':    [],
        'MSE_train':   [], 'MSE_val':    []}
optimizer_state_dicts = []
epochs_nb = []

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_mse, train_epoch_mae = fit(train_loader)
    val_epoch_loss, val_epoch_mse, val_epoch_mae = validate(val_loader)

    # Keep track of Hubert loss 
    writer.add_scalars(main_tag="Huber_Loss", 
                       tag_scalar_dict={"train_loss": train_epoch_loss,
                                            "test_loss": val_epoch_loss},
                       global_step=epoch)
    loss['Hubert_train'].append(train_epoch_loss)
    loss['Hubert_val'].append(val_epoch_loss)

    # Keep track of MSE loss
    writer.add_scalars(main_tag="MSE", 
                       tag_scalar_dict={"train_MSE": train_epoch_mse,
                                            "test_MSE": val_epoch_mse},
                      global_step=epoch) 
    loss['MSE_train'].append(train_epoch_mse)
    loss['MSE_val'].append(val_epoch_mse)

    # Keep track of MAE Loss 
    writer.add_scalars(main_tag="MAE", 
                       tag_scalar_dict={"train_MAE": train_epoch_mae,
                                            "test_MAE": val_epoch_mae},
                      global_step=epoch)        
    loss['MAE_train'].append(train_epoch_mse)
    loss['MAE_val'].append(val_epoch_mse)

    # update EarlyStopping class
    early_stopping(val_epoch_loss)
    if early_stopping.counter == 0:
    # save a checkpoint if the validation loss has improved for this epoch
        save_checkpoint(epoch, val_epoch_loss, timestamp, model, optimizer)
    
    # save an history
    optimizer_state_dicts.append(optimizer.state_dict)
    epochs_nb.append(epoch)
    save_history(epochs_nb, optimizer_state_dicts, loss, timestamp)
    
    print(f"Train Loss: {train_epoch_loss:.4f}, Train MSE: {train_epoch_mse:.2f}")
    print(f'Validation Loss: {val_epoch_loss:.4f}, Validation MSE: {val_epoch_mse:.2f}')
    if early_stopping.early_stop:
        break
end = time.time()
print(f"Training time: {(end-start)/60:.3f} minutes")
writer.flush()