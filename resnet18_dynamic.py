# ### Required Libraries ###
# Importing standard libraries and PyTorch-related libraries for building and training deep learning models.

import time
import numpy as np
import torch
import logging
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from abc import ABCMeta, abstractmethod
import math
from tqdm import tqdm

# ### Initial Setup ###

# Logging configuration to save training results.
logging.basicConfig(
    filename='resnet18_cifar100_dynamic.log',
    force=True,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# A class to manage hyperparameters and execution settings.
class Arguments:
    """
    A class to hold arguments and hyperparameters for the experiment.
    This is used for simplicity in notebook environments instead of argparse.
    """
    def __init__(self):
        self.batch_size = 64          # Batch size for training
        self.test_batch_size = 16     # Batch size for testing
        self.epochs = 30              # Number of communication rounds
        self.lr = 0.1                 # Learning rate
        self.client_count = 10        # Number of clients
        self.E = 1                    # Number of local epochs
        self.alpha = 0.01             # Penalty parameter for the ADMM algorithm (reduced from 1.0)
        self.rp_dim = 10              # Initial dimension of the random projection space (for FedRP)
        
        # Dynamic projection parameters
        self.rp_dim_min = 10          # Minimum projection dimension
        self.rp_dim_max = 200         # Maximum projection dimension (reduced from 1000)
        self.rp_growth_rate = 10      # Linear growth rate per epoch (increased for faster growth)
        
        # Adaptive projection parameters
        self.adaptive_threshold_high = 0.5   # High change threshold
        self.adaptive_threshold_low = 0.1    # Low change threshold
        self.adaptive_increment = 50         # Dimension increment when converging
        self.adaptive_decrement = 20         # Dimension decrement when diverging

args = Arguments()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Print device information
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("CUDA not available, using CPU")
print()

# ### Data Preparation ###

def get_datasets():
    """
    Downloads and prepares the CIFAR-100 dataset with necessary transforms.
    Transforms include normalization and data augmentation like random horizontal flipping.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=test_transform, download=True)
    return train_dataset, test_dataset

class FederatedDataset(Dataset):
    """
    A class to partition a standard dataset among multiple clients.
    This class ensures that data is distributed in a balanced (IID) manner,
    where each client receives a portion of data from all classes.
    """
    def __init__(self, dataset: Dataset, num_clients: int, client_id: int):
        super(FederatedDataset, self).__init__()
        self.dataset = dataset
        self.client_id = client_id
        self.num_clients = num_clients
        # Distribute data balancedly among clients
        client_data_indices = self._distribute_data_balanced()
        self.map = client_data_indices[client_id]
        np.random.shuffle(self.map)
        self.len = len(self.map)

    def _distribute_data_balanced(self):
        """
        Distributes data indices balancedly based on class labels among clients.
        """
        num_samples = len(self.dataset)
        targets = np.array([self.dataset[i][1] for i in range(num_samples)])
        classes, _ = np.unique(targets, return_counts=True)
        class_indices = {cls: np.where(targets == cls)[0] for cls in classes}
        
        client_data_indices = [[] for _ in range(self.num_clients)]
        for indices in class_indices.values():
            np.random.shuffle(indices)
            splits = np.array_split(indices, self.num_clients)
            for cid, split in enumerate(splits):
                client_data_indices[cid].extend(split)
        return client_data_indices

    def __getitem__(self, index):
        return self.dataset[self.map[index]]

    def __len__(self):
        return self.len

class NonIIDFederatedDataset(Dataset):
    """
    A class to partition a dataset among multiple clients with Non-IID distribution.
    Uses Dirichlet distribution to create heterogeneous data partitions.
    """
    def __init__(self, dataset: Dataset, num_clients: int, client_id: int, alpha: float = 0.5):
        super(NonIIDFederatedDataset, self).__init__()
        self.dataset = dataset
        self.client_id = client_id
        self.num_clients = num_clients
        self.alpha = alpha  # Dirichlet concentration parameter (smaller = more non-IID)
        
        # Distribute data with Non-IID pattern
        client_data_indices = self._distribute_data_non_iid()
        self.map = client_data_indices[client_id]
        np.random.shuffle(self.map)
        self.len = len(self.map)

    def _distribute_data_non_iid(self):
        """
        Distributes data indices with Non-IID pattern using Dirichlet distribution.
        """
        num_samples = len(self.dataset)
        targets = np.array([self.dataset[i][1] for i in range(num_samples)])
        classes = np.unique(targets)
        num_classes = len(classes)
        
        client_data_indices = [[] for _ in range(self.num_clients)]
        
        # For each class, distribute samples according to Dirichlet distribution
        for cls in classes:
            cls_indices = np.where(targets == cls)[0]
            np.random.shuffle(cls_indices)
            
            # Sample proportions from Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
            proportions = proportions / proportions.sum()
            
            # Split indices according to proportions
            splits = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
            client_splits = np.split(cls_indices, splits)
            
            for cid, split in enumerate(client_splits):
                client_data_indices[cid].extend(split)
        
        return client_data_indices

    def __getitem__(self, index):
        return self.dataset[self.map[index]]

    def __len__(self):
        return self.len

# ### Model Definition ###

class ResNet18(nn.Module):
    """
    ResNet-18 model optimized for the CIFAR-100 dataset.
    Modifications include:
    1. Using a ResNet-18 model without pre-trained weights.
    2. Changing the first convolutional layer to accept 32x32 images.
    3. Removing the MaxPool layer to preserve spatial feature dimensions.
    4. Adjusting the final fully connected layer for 100 output classes.
    """
    def __init__(self, num_classes=100):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# ### Helper Functions ###

def evaluate_model(device, model, dataloader, criterion):
    """
    Evaluates the model's performance on the test dataset.
    Calculates and returns the model's loss and accuracy.
    """
    model.eval()
    loss, total, correct = 0, 0, 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    avg_loss = loss / len(dataloader.dataset)
    return avg_loss, accuracy

# ### Base Classes for Federated Learning ###

class FederatedLearning(metaclass=ABCMeta):
    """
    Abstract Base Class for Federated Learning algorithms.
    Defines the main methods that every FL algorithm should implement.
    """
    @abstractmethod
    def __init__(self, Model, device, client_count, optimizer, criterion):
        pass
    
    @abstractmethod
    def _client_update(self, client_id, lr, E):
        pass

    @abstractmethod
    def _server_aggregate(self):
        pass

    @abstractmethod
    def global_update(self, state, lr, E):
        pass
    
    def get_communication_cost(self):
        """
        Returns the communication cost for the current round.
        Should be implemented by subclasses.
        """
        return 0

class FLBase(FederatedLearning):
    """
    A base class for different Federated Learning implementations.
    Contains shared logic like sending the model to clients and setting up federated data.
    """
    def __init__(self, Model, device, client_count, optimizer, criterion):
        self.Model = Model
        self.device = device
        self.client_count = client_count
        # Create a model instance for each client
        self.models = [Model().to(self.device) for _ in range(self.client_count)]
        self.optimizer = optimizer
        self.criterion = criterion
        
        # Communication cost tracking
        self.total_communication_cost = 0
        self.round_communication_costs = []

    def _send_model(self, state):
        """
        Sends the global model state (parameters) to all clients.
        """
        for model in self.models:
            model.load_state_dict(state.copy())
        # Reset weights and metrics for each round
        self.weights = [0] * self.client_count
        self.losses = [0] * self.client_count
        self.accuracies = [0] * self.client_count

    def setup_federated_data(self, dataset, batch_size, non_iid=False, alpha=0.5):
        """
        Creates a federated dataloader for each client.
        """
        if non_iid:
            self.client_dataloaders = [
                DataLoader(NonIIDFederatedDataset(dataset, self.client_count, cid, alpha), 
                          batch_size=batch_size, shuffle=True)
                for cid in range(self.client_count)
            ]
        else:
            self.client_dataloaders = [
                DataLoader(FederatedDataset(dataset, self.client_count, cid), 
                          batch_size=batch_size, shuffle=True)
                for cid in range(self.client_count)
            ]
    
    def _calculate_communication_cost(self, num_params):
        """
        Calculates the communication cost for the current round.
        Cost is measured as the number of parameters transmitted.
        """
        # Each client uploads num_params, server broadcasts num_params
        # Total = client_count * num_params (upload) + num_params (download per client)
        # Simplified: client_count * num_params (only counting uploads)
        return self.client_count * num_params
    
    def get_communication_cost(self):
        """
        Returns the total communication cost so far.
        """
        return self.total_communication_cost

# ### FedAvg Algorithm Implementation ###
# This algorithm is used as a baseline for comparison.

class FedAvg(FLBase):
    """
    Implementation of the Federated Averaging (FedAvg) algorithm.
    """
    def __init__(self, Model, device, client_count, optimizer, criterion):
        super(FedAvg, self).__init__(Model, device, client_count, optimizer, criterion)
        self.num_params = sum(p.numel() for p in Model().parameters())
        
    def _client_update(self, client_id, lr, E):
        """
        Trains the model on a client's local data.
        """
        model = self.models[client_id]
        optimizer = self.optimizer(model.parameters(), lr=lr)
        criterion = self.criterion()
        dataloader = self.client_dataloaders[client_id]
        
        weight, losses, total, correct = 0, 0, 0, 0
        for _ in range(E):
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                losses += loss.item()
                weight += len(data)
                optimizer.step()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
        self.weights[client_id] = weight / E
        self.losses[client_id] = losses / (E * weight) if weight > 0 else 0
        self.accuracies[client_id] = 100 * correct / total if total > 0 else 0

    def _server_aggregate(self):
        """
        Aggregates client model parameters on the server using a weighted average.
        """
        total_weight = sum(self.weights)
        self.normalized_weights = np.array(self.weights) / total_weight if total_weight > 0 else [1/self.client_count]*self.client_count
        
        with torch.no_grad():
            clients_updates = [model.state_dict() for model in self.models]
            aggregated_params = clients_updates[0].copy()
            
            for name in aggregated_params:
                # Only initialize and aggregate parameters that are floating point types
                if aggregated_params[name].dtype.is_floating_point:
                    aggregated_params[name] = torch.zeros_like(aggregated_params[name]).to(self.device)
            
            for client_id, params in enumerate(clients_updates):
                weight = float(self.normalized_weights[client_id])
                for name in aggregated_params:
                    # Only aggregate parameters that are floating point types
                    if aggregated_params[name].dtype.is_floating_point:
                        aggregated_params[name] += params[name].to(aggregated_params[name].dtype) * weight
                    
        return aggregated_params.copy()

    def global_update(self, state, lr, E=1, epoch=None):
        """
        Executes one full round of federated learning (send model, local training, aggregate).
        """
        self._send_model(state)
        for i in tqdm(range(self.client_count), desc="Client Updates", leave=False, unit="client"):
            self._client_update(i, lr, E)
        
        # Calculate communication cost
        round_cost = self._calculate_communication_cost(self.num_params)
        self.total_communication_cost += round_cost
        self.round_communication_costs.append(round_cost)
        
        avg_loss = sum(self.losses) / self.client_count
        avg_acc = sum(self.accuracies) / self.client_count
        return self._server_aggregate(), avg_loss, avg_acc

# ### FedAvg+DP Algorithm Implementation ###
# This algorithm adds Gaussian noise to the model parameters to provide differential privacy.

def clip_weights(weights, C):
    """
    Clips gradients to limit the influence of a single data point.
    """
    norm = torch.norm(weights, p=2)
    scale = max(1.0, norm / C)
    return weights / scale

def add_dp_noise(weights, C, sigma):
    """
    Adds Gaussian noise to achieve differential privacy.
    """
    noise = torch.normal(0, sigma * C, size=weights.size(), device=weights.device)
    return weights + noise

class FedAvgDP(FedAvg):
    """
    Implements FedAvg with Differential Privacy (DP).
    After local training, parameters are clipped and noise is added.
    """
    def __init__(self, Model, device, client_count, optimizer, criterion):
        super(FedAvgDP, self).__init__(Model, device, client_count, optimizer, criterion)
        self.C = None      # No clipping (set to None to disable)
        self.sigma = 0.1   # Noise level σ = 0.1

    def _client_update(self, client_id, lr, E):
        # First, perform standard local training
        super()._client_update(client_id, lr, E)
        
        # Apply DP mechanisms (only noise, no clipping)
        model = self.models[client_id]
        for name, param in model.named_parameters():
            if 'weight' in name:
                # No clipping, only add noise
                if self.C is not None:
                    param.data = clip_weights(param.data, self.C)
                param.data = add_dp_noise(param.data, 1.0, self.sigma)  # Use C=1.0 for noise scale

# ### ADMM (FedADMM) Algorithm Implementation ###
# This algorithm uses the ADMM method to achieve consensus among clients.

class FedADMM(FLBase):
    """
    Implementation of Federated Learning using the ADMM optimization framework.
    """
    def __init__(self, Model, device, client_count, optimizer, criterion, alpha):
        super(FedADMM, self).__init__(Model, device, client_count, optimizer, criterion)
        self.alpha = alpha  # Penalty parameter
        self.num_params = sum(p.numel() for p in Model().parameters())
        
        with torch.no_grad():
            self.server_model = Model().to(self.device)  # z variable or server model
        
        params = list(self.Model().parameters())
        # Dual variable y
        self.y = [[torch.zeros_like(p).to(self.device) for p in params] for _ in range(self.client_count)]
        self.epoch_tracker = [0] * self.client_count

    def _update_y(self, client_id):
        """
        Updates the dual variable y for each client.
        """
        if self.epoch_tracker[client_id] != 0:
            with torch.no_grad():
                client_params = list(self.models[client_id].parameters())
                server_params = list(self.server_model.parameters())
                for i in range(len(self.y[client_id])):
                    self.y[client_id][i] += self.alpha * (client_params[i] - server_params[i])
        else:
            self.epoch_tracker[client_id] = 1

    def _client_update(self, client_id, lr, E):
        """
        Performs local client training with the modified ADMM cost function.
        """
        self._update_y(client_id)
        
        model = self.models[client_id]
        optimizer = self.optimizer(model.parameters(), lr=lr)
        criterion = self.criterion()
        dataloader = self.client_dataloaders[client_id]
        
        weight, losses, total, correct = 0, 0, 0, 0
        for _ in range(E):
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                
                # Calculate the main loss
                loss = criterion(output, target)
                losses += loss.item()
                
                # Add ADMM penalty and dual terms to the loss
                # NOTE: Scale down the ADMM terms to not overwhelm the classification loss
                client_params_flat = torch.cat([p.flatten() for p in model.parameters()])
                server_params_flat = torch.cat([p.flatten() for p in self.server_model.parameters()]).detach()
                y_flat = torch.cat([y_i.flatten() for y_i in self.y[client_id]]).detach()
                
                admm_penalty = (self.alpha / 2) * torch.norm(client_params_flat - server_params_flat) ** 2
                admm_dual = torch.dot(y_flat, client_params_flat - server_params_flat)
                
                # Scale ADMM terms to be comparable to classification loss (which is ~5.0 for CIFAR-100)
                loss += admm_penalty / len(client_params_flat)
                loss += admm_dual / len(client_params_flat)
                
                loss.backward()
                optimizer.step()
                weight += len(data)

                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        self.weights[client_id] = weight / E
        self.losses[client_id] = losses / (E * weight) if weight > 0 else 0
        self.accuracies[client_id] = 100 * correct / total if total > 0 else 0

    def _server_aggregate(self):
        """
        Aggregates client parameters to update the global model (z).
        """
        total_weight = sum(self.weights)
        normalized_weights = np.array(self.weights) / total_weight if total_weight > 0 else [1/self.client_count]*self.client_count
        
        with torch.no_grad():
            clients_updates = [model.state_dict() for model in self.models]
            aggregated_params = clients_updates[0].copy()
            
            for name in aggregated_params:
                # Only initialize and aggregate parameters that are floating point types
                if aggregated_params[name].dtype.is_floating_point:
                    aggregated_params[name] = torch.zeros_like(aggregated_params[name]).to(self.device)
            
            for cid, params in enumerate(clients_updates):
                weight = float(normalized_weights[cid])
                for name in aggregated_params:
                    # Only aggregate parameters that are floating point types
                    if aggregated_params[name].dtype.is_floating_point:
                        aggregated_params[name] += params[name].to(aggregated_params[name].dtype) * weight
                    
        return aggregated_params.copy()

    def global_update(self, state, lr, E=1, epoch=None):
        """
        Executes one full round of FedADMM.
        """
        self.server_model.load_state_dict(state.copy())
        # Reset metrics for the round
        self.weights = [0] * self.client_count
        self.losses = [0] * self.client_count
        self.accuracies = [0] * self.client_count
        
        for i in tqdm(range(self.client_count), desc="Client Updates", leave=False, unit="client"):
            self._client_update(i, lr, E)
        
        # Calculate communication cost
        round_cost = self._calculate_communication_cost(self.num_params)
        self.total_communication_cost += round_cost
        self.round_communication_costs.append(round_cost)
        
        avg_loss = sum(self.losses) / self.client_count
        avg_acc = sum(self.accuracies) / self.client_count
        aggregated_state = self._server_aggregate()
        self.server_model.load_state_dict(aggregated_state)
        
        return aggregated_state, avg_loss, avg_acc
        
# ### FedRP Algorithm Implementation (Main method from the paper) ###
# This algorithm combines ADMM with random projection to simultaneously improve privacy
# and reduce communication cost.

class FedRP(FLBase):
    """
    Implementation of the FedRP algorithm combining ADMM and random projection.
    """
    def __init__(self, Model, device, client_count, optimizer, criterion, alpha, rp_dim):
        super(FedRP, self).__init__(Model, device, client_count, optimizer, criterion)
        self.alpha = alpha
        self.rp_dim = rp_dim
        self.num_params = sum(p.numel() for p in Model().parameters())
        
        # ADMM variables in the projected space
        self.server_z = torch.zeros(self.rp_dim).to(self.device)
        self.y = [torch.zeros(self.rp_dim).to(self.device) for _ in range(self.client_count)]
        self.epoch_tracker = [0] * self.client_count

    def _generate_projection_matrix(self, seed):
        """
        Creates a shared random projection matrix for all clients in a round.
        This matrix is unknown to the server.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        # Using a variance of 1/n for computational stability
        return (torch.randn(self.rp_dim, self.num_params) / math.sqrt(self.num_params)).to(self.device).detach()

    def _update_y(self, client_id, proj_matrix):
        """
        Updates the dual variable y in the projected space.
        """
        if self.epoch_tracker[client_id] != 0:
            with torch.no_grad():
                client_params_flat = torch.cat([p.flatten() for p in self.models[client_id].parameters()])
                client_z = torch.matmul(proj_matrix, client_params_flat).detach()
                self.y[client_id] += self.alpha * (client_z - self.server_z)
        else:
            self.epoch_tracker[client_id] = 1

    def _client_update(self, client_id, lr, E, proj_matrix):
        """
        Performs local client training with the modified FedRP cost function.
        The loss is calculated based on projected parameters.
        """
        self._update_y(client_id, proj_matrix)
        
        model = self.models[client_id]
        optimizer = self.optimizer(model.parameters(), lr=lr)
        criterion = self.criterion()
        dataloader = self.client_dataloaders[client_id]
        
        weight, losses, total, correct = 0, 0, 0, 0
        for _ in range(E):
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                
                loss = criterion(output, target)
                losses += loss.item()
                
                # Add ADMM penalty and dual terms in the projected space (Equation 9 from the paper)
                # NOTE: Scale down to not overwhelm the classification loss
                client_params_flat = torch.cat([p.flatten() for p in model.parameters()])
                # Detach projection matrix to avoid gradient backprop through it
                client_z = torch.matmul(proj_matrix.detach(), client_params_flat)
                
                admm_penalty = (self.alpha / 2) * torch.norm(client_z - self.server_z.detach()) ** 2
                admm_dual = torch.dot(self.y[client_id].detach(), client_z - self.server_z.detach())
                
                # Scale by projection dimension to normalize
                loss += admm_penalty / self.rp_dim
                loss += admm_dual / self.rp_dim
                
                loss.backward()
                optimizer.step()
                weight += len(data)

                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        # Store projected parameters for aggregation
        with torch.no_grad():
            client_params_flat = torch.cat([p.flatten() for p in self.models[client_id].parameters()])
            self.projected_params[client_id] = torch.matmul(proj_matrix, client_params_flat).detach()
        self.weights[client_id] = weight / E
        self.losses[client_id] = losses / (E * weight) if weight > 0 else 0
        self.accuracies[client_id] = 100 * correct / total if total > 0 else 0

    def _server_aggregate(self):
        """
        Aggregates the projected vectors z_i to update the global z_bar.
        """
        total_weight = sum(self.weights)
        normalized_weights = np.array(self.weights) / total_weight if total_weight > 0 else [1/self.client_count]*self.client_count
        
        with torch.no_grad():
            aggregated_z = torch.zeros(self.rp_dim).to(self.device)
            for cid in range(self.client_count):
                aggregated_z += self.projected_params[cid] * normalized_weights[cid]
        return aggregated_z.clone()

    def global_update(self, state, lr, E, epoch):
        """
        Executes one full round of the FedRP algorithm.
        """
        self._send_model(state) # Send the model w to clients
        # Reset metrics for the round
        self.weights = [0] * self.client_count
        self.losses = [0] * self.client_count
        self.accuracies = [0] * self.client_count
        self.projected_params = [None] * self.client_count
        
        # Generate a new projection matrix for each round
        proj_matrix = self._generate_projection_matrix(seed=42 + epoch)
        
        for i in tqdm(range(self.client_count), desc="Client Updates", leave=False, unit="client"):
            self._client_update(i, lr, E, proj_matrix)
        
        # Aggregate and update z_bar
        self.server_z = self._server_aggregate()
        
        # Calculate communication cost (only projected dimensions are transmitted)
        round_cost = self._calculate_communication_cost(self.rp_dim)
        self.total_communication_cost += round_cost
        self.round_communication_costs.append(round_cost)
        
        # Since the server does not have access to the original parameters (w),
        # for evaluation, we consider one client's model (e.g., the last one) as the global model.
        final_state = self.models[-1].state_dict().copy()
        avg_loss = sum(self.losses) / self.client_count
        avg_acc = sum(self.accuracies) / self.client_count
        
        return final_state, avg_loss, avg_acc

# ### Dynamic FedRP Base Class ###
# Base class for dynamic projection dimension strategies

class DynamicFedRP(FedRP):
    """
    Base class for FedRP with dynamic projection dimension.
    Subclasses implement different strategies for adjusting m(t).
    """
    def __init__(self, Model, device, client_count, optimizer, criterion, alpha, rp_dim_min, rp_dim_max):
        # Initialize with minimum dimension
        super(DynamicFedRP, self).__init__(Model, device, client_count, optimizer, criterion, alpha, rp_dim_min)
        self.rp_dim_min = rp_dim_min
        self.rp_dim_max = rp_dim_max
        self.current_rp_dim = rp_dim_min
        
        # Track dimension changes over epochs
        self.dimension_history = []
        self.previous_z = None
        
    def _update_projection_dimension(self, epoch):
        """
        Updates the projection dimension for the current epoch.
        To be implemented by subclasses.
        """
        raise NotImplementedError
    
    def _resize_admm_variables(self, new_dim):
        """
        Resizes ADMM variables (z and y) when projection dimension changes.
        """
        old_dim = self.rp_dim
        self.rp_dim = new_dim
        
        with torch.no_grad():
            # Resize server_z
            if new_dim > old_dim:
                # Expand with zeros
                new_z = torch.zeros(new_dim).to(self.device)
                new_z[:old_dim] = self.server_z
                self.server_z = new_z
            else:
                # Truncate
                self.server_z = self.server_z[:new_dim]
            
            # Resize dual variables y for all clients
            for i in range(self.client_count):
                if new_dim > old_dim:
                    new_y = torch.zeros(new_dim).to(self.device)
                    new_y[:old_dim] = self.y[i]
                    self.y[i] = new_y
                else:
                    self.y[i] = self.y[i][:new_dim]
    
    def global_update(self, state, lr, E, epoch):
        """
        Executes one full round with dynamic projection dimension.
        """
        # Store previous z for adaptive strategy
        if self.server_z is not None:
            self.previous_z = self.server_z.clone()
        
        # Update projection dimension based on strategy
        new_dim = self._update_projection_dimension(epoch)
        if new_dim != self.current_rp_dim:
            self._resize_admm_variables(new_dim)
            self.current_rp_dim = new_dim
        
        self.dimension_history.append(self.current_rp_dim)
        
        # Call parent's global_update with current dimension
        return super().global_update(state, lr, E, epoch)

# ### Linear Growth Strategy ###

class FedRP_Linear(DynamicFedRP):
    """
    FedRP with linear growth of projection dimension.
    m(t) = min(m_min + α * t, m_max)
    """
    def __init__(self, Model, device, client_count, optimizer, criterion, alpha, 
                 rp_dim_min, rp_dim_max, growth_rate):
        super(FedRP_Linear, self).__init__(Model, device, client_count, optimizer, criterion, 
                                           alpha, rp_dim_min, rp_dim_max)
        self.growth_rate = growth_rate
    
    def _update_projection_dimension(self, epoch):
        """
        Linear growth: m(t) = min(m_min + growth_rate * t, m_max)
        """
        new_dim = int(min(self.rp_dim_min + self.growth_rate * epoch, self.rp_dim_max))
        return new_dim

# ### Adaptive Strategy ###

class FedRP_Adaptive(DynamicFedRP):
    """
    FedRP with adaptive adjustment of projection dimension based on model convergence.
    Monitors the change in global consensus vector z to adjust dimension.
    """
    def __init__(self, Model, device, client_count, optimizer, criterion, alpha, 
                 rp_dim_min, rp_dim_max, threshold_high, threshold_low, 
                 increment, decrement):
        super(FedRP_Adaptive, self).__init__(Model, device, client_count, optimizer, criterion, 
                                            alpha, rp_dim_min, rp_dim_max)
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.increment = increment
        self.decrement = decrement
    
    def _update_projection_dimension(self, epoch):
        """
        Adaptive adjustment based on ||z_t - z_{t-1}||.
        If change is small (converging), increase dimension to capture fine details.
        If change is large (still training), can maintain or decrease dimension.
        """
        if self.previous_z is None or epoch == 0:
            # First epoch, keep minimum dimension
            return self.current_rp_dim
        
        # Calculate the change in z (normalized by dimension)
        min_dim = min(len(self.server_z), len(self.previous_z))
        z_change = torch.norm(self.server_z[:min_dim] - self.previous_z[:min_dim]).item()
        z_norm = torch.norm(self.previous_z[:min_dim]).item()
        
        if z_norm > 0:
            relative_change = z_change / z_norm
        else:
            relative_change = z_change
        
        new_dim = self.current_rp_dim
        
        # If change is small (converging), increase dimension
        if relative_change < self.threshold_low:
            new_dim = min(self.current_rp_dim + self.increment, self.rp_dim_max)
        # If change is large (diverging), decrease dimension to save communication
        elif relative_change > self.threshold_high:
            new_dim = max(self.current_rp_dim - self.decrement, self.rp_dim_min)
        
        return int(new_dim)

# ### Main Experiment Runner Function ###

def run_experiment(algorithm_class, train_dataset, test_dataset, args, 
                   algorithm_name=None, non_iid=False, **kwargs):
    """
    Runs a complete experiment for a specified Federated Learning algorithm.
    """
    if algorithm_name is None:
        algorithm_name = algorithm_class.__name__
    
    data_type = "Non-IID" if non_iid else "IID"
    print(f"\n--- Running Experiment: {algorithm_name} with {args.client_count} clients ({data_type}) ---")
    
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    # Instantiate the federated algorithm
    fl_instance = algorithm_class(ResNet18, device, args.client_count, 
                                  optim.SGD, nn.CrossEntropyLoss, **kwargs)
    fl_instance.setup_federated_data(train_dataset, args.batch_size, non_iid=non_iid)
    
    # Initial global model
    global_model = ResNet18().to(device)
    global_state = global_model.state_dict().copy()
    criterion = nn.CrossEntropyLoss()
    
    # Tracking metrics
    test_accuracies = []
    convergence_epoch = None
    target_accuracy = 30.0  # Define a target accuracy for convergence
    
    start_time = time.time()
    for epoch in tqdm(range(args.epochs), desc=f"{algorithm_name} Training", unit="epoch"):
        # Perform one global update round
        global_state, train_loss, train_acc = fl_instance.global_update(
            global_state, lr=args.lr, E=args.E, epoch=epoch)
        
        # Evaluate the global model
        global_model.load_state_dict(global_state.copy())
        test_loss, test_acc = evaluate_model(device, global_model, test_loader, criterion)
        test_accuracies.append(test_acc)
        
        # Check for convergence
        if convergence_epoch is None and test_acc >= target_accuracy:
            convergence_epoch = epoch + 1
        
        # Log dimension for dynamic methods
        dimension_info = ""
        if hasattr(fl_instance, 'current_rp_dim'):
            dimension_info = f" | RP Dim: {fl_instance.current_rp_dim}"
        
        tqdm.write(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {test_acc:.2f}%{dimension_info}")
        
        logging.info(f"Epoch {epoch+1}, client_count {args.client_count}, "
                    f"Algo: {algorithm_name}, data_type: {data_type}: "
                    f"train_accuracy={train_acc:.4f}, test_accuracy={test_acc:.4f}, "
                    f"train_loss={train_loss:.4f}, test_loss={test_loss:.4f}{dimension_info}")
    
    training_time = time.time() - start_time
    total_comm_cost = fl_instance.get_communication_cost()
    final_accuracy = test_accuracies[-1]
    
    # Summary statistics
    print(f"\n=== Summary for {algorithm_name} ({data_type}) ===")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Final test accuracy: {final_accuracy:.2f}%")
    print(f"Total communication cost: {total_comm_cost:.2e} parameters")
    if convergence_epoch:
        print(f"Converged at epoch: {convergence_epoch}")
        print(f"Communication cost to convergence: {sum(fl_instance.round_communication_costs[:convergence_epoch]):.2e} parameters")
    
    # Log dimension history for dynamic methods
    if hasattr(fl_instance, 'dimension_history'):
        print(f"Dimension range: {min(fl_instance.dimension_history)} - {max(fl_instance.dimension_history)}")
        logging.info(f"Algo: {algorithm_name}, Dimension history: {fl_instance.dimension_history}")
    
    logging.info(f"=== Summary for {algorithm_name} ({data_type}) ===")
    logging.info(f"Training time: {training_time:.2f}s, Final accuracy: {final_accuracy:.2f}%, "
                f"Total comm cost: {total_comm_cost:.2e}, Convergence epoch: {convergence_epoch}")
    
    return {
        'algorithm': algorithm_name,
        'data_type': data_type,
        'final_accuracy': final_accuracy,
        'training_time': training_time,
        'total_comm_cost': total_comm_cost,
        'convergence_epoch': convergence_epoch,
        'test_accuracies': test_accuracies,
        'dimension_history': getattr(fl_instance, 'dimension_history', None)
    }

# ### Program Entry Point ###

if __name__ == '__main__':
    train_data, test_data = get_datasets()
    
    results = []
    
    print("\n" + "="*80)
    print("EXPERIMENTS WITH IID DATA")
    print("="*80)
    
    # # ✅ COMPLETED - Baseline: FedAvg
    # results.append(run_experiment(
    #     FedAvg, train_data, test_data, args,
    #     algorithm_name="FedAvg"
    # ))
    
    # # ✅ COMPLETED - Baseline: FedAvg with DP
    # results.append(run_experiment(
    #     FedAvgDP, train_data, test_data, args,
    #     algorithm_name="FedAvgDP"
    # ))
    
    # # ✅ COMPLETED - Baseline: FedADMM
    # results.append(run_experiment(
    #     FedADMM, train_data, test_data, args,
    #     algorithm_name="FedADMM",
    #     alpha=args.alpha
    # ))
    
    # # ✅ COMPLETED - Original FedRP with fixed dimension (small)
    # results.append(run_experiment(
    #     FedRP, train_data, test_data, args,
    #     algorithm_name="FedRP (m=10)",
    #     alpha=args.alpha, rp_dim=10
    # ))
    
    # # ⏸️ CONTINUE FROM HERE - Original FedRP with fixed dimension (medium)
    # results.append(run_experiment(
    #     FedRP, train_data, test_data, args,
    #     algorithm_name="FedRP (m=100)",
    #     alpha=args.alpha, rp_dim=100
    # ))
    
    # # Original FedRP with fixed dimension (large)
    # results.append(run_experiment(
    #     FedRP, train_data, test_data, args,
    #     algorithm_name="FedRP (m=1000)",
    #     alpha=args.alpha, rp_dim=1000
    # ))
    
    # NEW: FedRP with Linear Growth
    # results.append(run_experiment(
    #     FedRP_Linear, train_data, test_data, args,
    #     algorithm_name="FedRP_Linear",
    #     alpha=args.alpha,
    #     rp_dim_min=args.rp_dim_min,
    #     rp_dim_max=args.rp_dim_max,
    #     growth_rate=args.rp_growth_rate
    # ))
    
    # NEW: FedRP with Adaptive Strategy
    results.append(run_experiment(
        FedRP_Adaptive, train_data, test_data, args,
        algorithm_name="FedRP_Adaptive",
        alpha=args.alpha,
        rp_dim_min=args.rp_dim_min,
        rp_dim_max=args.rp_dim_max,
        threshold_high=args.adaptive_threshold_high,
        threshold_low=args.adaptive_threshold_low,
        increment=args.adaptive_increment,
        decrement=args.adaptive_decrement
    ))
    
    print("\n" + "="*80)
    print("EXPERIMENTS WITH NON-IID DATA")
    print("="*80)
    
    # Run key comparisons with Non-IID data
    # results.append(run_experiment(
    #     FedAvg, train_data, test_data, args,
    #     algorithm_name="FedAvg",
    #     non_iid=True
    # ))
    
    # results.append(run_experiment(
    #     FedRP, train_data, test_data, args,
    #     algorithm_name="FedRP (m=100)",
    #     alpha=args.alpha, rp_dim=100,
    #     non_iid=True
    # ))
    
    # results.append(run_experiment(
    #     FedRP_Linear, train_data, test_data, args,
    #     algorithm_name="FedRP_Linear",
    #     alpha=args.alpha,
    #     rp_dim_min=args.rp_dim_min,
    #     rp_dim_max=args.rp_dim_max,
    #     growth_rate=args.rp_growth_rate,
    #     non_iid=True
    # ))
    
    # results.append(run_experiment(
    #     FedRP_Adaptive, train_data, test_data, args,
    #     algorithm_name="FedRP_Adaptive",
    #     alpha=args.alpha,
    #     rp_dim_min=args.rp_dim_min,
    #     rp_dim_max=args.rp_dim_max,
    #     threshold_high=args.adaptive_threshold_high,
    #     threshold_low=args.adaptive_threshold_low,
    #     increment=args.adaptive_increment,
    #     decrement=args.adaptive_decrement,
    #     non_iid=True
    # ))
    
    # Print final comparison table
    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)
    print(f"{'Algorithm':<25} {'Data':<10} {'Accuracy':<12} {'Comm Cost':<15} {'Conv Epoch':<12}")
    print("-"*80)
    for r in results:
        comm_cost_str = f"{r['total_comm_cost']:.2e}"
        conv_epoch_str = str(r['convergence_epoch']) if r['convergence_epoch'] else "N/A"
        print(f"{r['algorithm']:<25} {r['data_type']:<10} {r['final_accuracy']:>10.2f}% "
              f"{comm_cost_str:>14} {conv_epoch_str:>11}")
    
    logging.info("\n" + "="*80)
    logging.info("FINAL RESULTS COMPARISON")
    logging.info("="*80)
    for r in results:
        logging.info(f"{r['algorithm']} ({r['data_type']}): Accuracy={r['final_accuracy']:.2f}%, "
                    f"CommCost={r['total_comm_cost']:.2e}, ConvEpoch={r['convergence_epoch']}")
