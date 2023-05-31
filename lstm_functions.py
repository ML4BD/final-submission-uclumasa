import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

class MasteryDataset(Dataset):
    def __init__(self, df, target_week=6):
        self.df = df
        self.target_week = target_week
        # Get the list of valid user IDs that have data for the target week
        self.user_ids = self._get_filtered_user_ids()

    def _get_filtered_user_ids(self):
        # Get unique user IDs in the dataset
        user_ids = self.df['user_id'].unique()
        valid_user_ids = []
        # Iterate through user IDs and check if they have data for the target week
        for user_id in user_ids:
            if self.df[(self.df['user_id'] == user_id) & (self.df['weeks_since_first_transaction'] == self.target_week)].shape[0] > 0:
                valid_user_ids.append(user_id)
        return valid_user_ids

    def __len__(self):
        # Return the total number of valid user IDs
        return len(self.user_ids)

    def __getitem__(self, idx):
        # Get the user ID at the given index
        user_id = self.user_ids[idx]
        # Retrieve the user's data for weeks before the target week, 
        # sorted by the week,
        # and drop unnecessary columns
        user_data = self.df[(self.df['user_id'] == user_id) & (self.df['weeks_since_first_transaction'] < self.target_week)].sort_values('weeks_since_first_transaction').drop(columns=['user_id', 'title'])
        
        # Pad the sequence with zeros
        pad_len = self.target_week - len(user_data)
        user_data = np.pad(user_data, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
        
        # Get input sequence (first 6 weeks) and target value (mastery at week 6)
        input_sequence = torch.tensor(user_data, dtype=torch.float32)
        target_value = torch.tensor(self.df.loc[(self.df['user_id'] == user_id) & (self.df['weeks_since_first_transaction'] == self.target_week), 'mastery'].values[0], dtype=torch.float32)
        
        return input_sequence, target_value


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(LSTMModel, self).__init__()
        # Set the hidden size, number of layers, and device for the model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        # Create an LSTM layer with the specified input size, hidden size, and number of layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(self.device)
        # Add a fully connected layer with the same input and output size as the hidden size
        self.fc1 = nn.Linear(hidden_size, hidden_size).to(self.device)  # Add an additional linear layer
        # Add a final fully connected layer to produce the output of the desired size
        self.fc2 = nn.Linear(hidden_size, output_size).to(self.device)

    def forward(self, x):
        # Initialize hidden state (h0) and cell state (c0) with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Pass the input through the LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        # Pass the output of the LSTM through the first fully connected layer
        out = self.fc1(out[:, -1, :])
        out = torch.relu(out)  # Add a ReLU activation function
        # Pass the output through the second fully connected layer to produce the final output
        out = self.fc2(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(GRUModel, self).__init__()
        # Set the hidden size, number of layers, and device for the model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        # Create an GRU layer with the specified input size, hidden size, and number of layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True).to(self.device)
         # Add a fully connected layer with the same input and output size as the hidden size
        self.fc1 = nn.Linear(hidden_size, hidden_size).to(self.device)  # Add an additional linear layer
        # Add a final fully connected layer to produce the output of the desired size
        self.fc2 = nn.Linear(hidden_size, output_size).to(self.device)

    def forward(self, x):
        # Initialize hidden state (h0) with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Pass the input through the GRU layer
        out, _ = self.gru(x, h0)
        # Pass the output of the GRU through the first fully connected layer
        out = self.fc1(out[:, -1, :])
        out = torch.relu(out)  # Add a ReLU activation function
        # Pass the output through the second fully connected layer to produce the final output
        out = self.fc2(out)
        return out