import torch

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def setup(self, output_labels, pad_token=0):
        self.output_labels = output_labels
        self.pad_token = pad_token

    def forward(self, log_probs: torch.Tensor):
        indices = torch.argmax(log_probs, dim=-1)

        predictions = []

        for p in list(indices):
            unique_indices = torch.unique_consecutive(p, dim=-1)
            prediction = "".join([ self.output_labels[t] for t in unique_indices if t != self.pad_token ])
            predictions.append(prediction)

        return predictions
