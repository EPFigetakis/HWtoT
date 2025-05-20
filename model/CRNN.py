import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, img_height, num_classes, cnn_out_channels=256, lstm_hidden=256, lstm_layers=2):
        super(CRNN, self).__init__()
        self.cnn = self._build_cnn(img_height, cnn_out_channels)
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=False  # Required for CTC Loss (T, B, C)
        )
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)  # bidirectional

    def _build_cnn(self, img_height, cnn_out_channels):
        # Output height must be fixed (e.g., 1 or 2) for the LSTM to scan across width
        layers = []

        def conv_bn_relu(in_c, out_c, kernel, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel, stride, padding),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        layers += [conv_bn_relu(1, 64, 3, 1, 1), nn.MaxPool2d(2, 2)]         # [H/2, W/2]
        layers += [conv_bn_relu(64, 128, 3, 1, 1), nn.MaxPool2d(2, 2)]       # [H/4, W/4]
        layers += [conv_bn_relu(128, 256, 3, 1, 1)]
        layers += [conv_bn_relu(256, cnn_out_channels, 3, 1, 1)]
        layers += [nn.AdaptiveAvgPool2d((1, None))]  # output shape = [B, C, 1, W]

        return nn.Sequential(*layers)

    def forward(self, x):
        # Input: x = [B, 1, H, W]
        x = self.cnn(x)              # [B, C, 1, W]
        x = x.squeeze(2)             # [B, C, W]
        x = x.permute(2, 0, 1)       # [W, B, C] → time-first for CTC

        x, _ = self.lstm(x)          # [W, B, 2*H]
        x = self.fc(x)               # [W, B, num_classes]

        return x