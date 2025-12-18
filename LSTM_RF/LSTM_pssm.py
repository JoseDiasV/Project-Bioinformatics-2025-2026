import torch
from torch import nn

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
class Softmax(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        """
        Softmax layer for base LSTM model
        - Not used for LSTM-RF
        """
        super().__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        pred = self.linear(x)
        return pred
