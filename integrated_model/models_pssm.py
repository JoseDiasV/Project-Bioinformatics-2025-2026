import torch

from torch import nn

import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes, window_size):

        """
        Building blocks of convolutional neural network.

        Parameters:
            * in_channels: Number of channels in the input image (for grayscale images, 1)
            * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
        """
        super(CNN, self).__init__()
        # Do you want to return ReLU insted of final vector?
        #self.ReLU_out = ReLU_out

        # 1st convolutional layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=500, kernel_size=5, padding=2)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2nd convolutional layer
        self.conv2 = nn.Conv2d(in_channels=500, out_channels=100, kernel_size=2, padding=1)
        # Fully connected layer
        #self.fc1 = nn.Linear(100 * 7 * 11, num_classes)
        self.fc1 = nn.Linear(1, num_classes)  # temporary
        self._initialize_fc(in_channels, num_classes, window_size)

    def _initialize_fc(self, in_channels, num_classes, wind_size):
        # Build a dummy input to compute final flattened size
        x = torch.zeros(1, in_channels, wind_size, 20)   # adjust if your input size differs
        x = self.pool(self.conv1(x))
        x = F.relu(self.conv2(x))
        flat_size = x.numel()

        # Redefine fc1 with correct size
        self.fc1 = nn.Linear(flat_size, num_classes)

    def forward(self, x, ReLU_out=False):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: Input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = self.conv1(x)          # Apply first convolution without activation layer
        x = self.pool(x)           # Apply max pooling
        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor into (batch_size,7700)

        if ReLU_out: # used only on trained CNN and in case we extract ReLU
            return x
        
        x = self.fc1(x)            # Apply fully connected layer
        return x
 
class Softmax(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        pred = self.linear(x)
        return pred 
    
class LSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        """
        LSTM architecture:

        For base LSTM model:
        - Includes a Softmax layer for direct classification.

        For LSTM-RF model:
        - The get_last_layer_features() method is used to extract features
          from the second (last) LSTM layer to feed a Random Forest classifier.
        """
        super(LSTM, self).__init__()

        # ===== First Layer: sequence LSTM =====
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=400,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # ===== Second Layer: last LSTM =====
        self.lstm2 = nn.LSTM(
            input_size=400 * 2,  # bidirectional output from lstm1
            hidden_size=800,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Output dimension after second layer
        lstm2_out_dim = 800 * 2  # bidirectional

        # ===== Third Layer: classification =====
        self.classifier = nn.Linear(lstm2_out_dim, num_classes)

        # ===== Fourth Layer: full connectivity (Softmax in base LSTM) =====
        self.fc = nn.Linear(num_classes, num_classes)

    def forward(self, x, lengths=None):
        """
        Forward pass for base LSTM:
        - x: (B, L, input_dim)
        - lengths: optional sequence lengths
        - returns: (B, L, num_classes)
        """
        # Pack sequences if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # ===== First LSTM layer =====
        out, _ = self.lstm1(x)

        # ===== Second LSTM layer =====
        out, _ = self.lstm2(out)

        # Unpack if packed
        if lengths is not None:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # ===== Third Layer: classification =====
        out = self.classifier(out)

        # ===== Fourth Layer: fully-connected (Softmax in base LSTM) =====
        out = self.fc(out)

        return out

    def get_last_layer_features(self, x):
        """
        Feature extractor for LSTM-RF:
        - Returns output of second (last) LSTM layer at last timestep
        - Used only for feeding a Random Forest classifier
        """
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        features = out[:, -1, :]  # (batch, 1600) bidirectional
        return features

# ===== Softmax layer (only for base LSTM) =====
# class Softmax(nn.Module):
#     def __init__(self, n_inputs, n_outputs):
#         """
#         Softmax layer for base LSTM model
#         - Not used for LSTM-RF
#         """
#         super().__init__()
#         self.linear = nn.Linear(n_inputs, n_outputs)

#     def forward(self, x):
#         pred = self.linear(x)
#         return pred