import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
import snntorch.functional as SF
import snntorch.utils as utils
import matplotlib.pyplot as plt


# Network Parameters
num_inputs = 100
num_hidden = 100
num_outputs = 10
beta = 0.9  # Decay rate for LIF neurons


class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)  # LIF neuron
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)  # LIF neuron

    def forward(self, x):
        mem1 = self.lif1.init_leaky()  # Initialize membrane potential
        mem2 = self.lif2.init_leaky()
        spk1, mem1 = self.lif1(self.fc1(x), mem1)
        spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
        return spk2, mem2  # Return spikes and membrane potential


def train():
    snn_model = SNN()
    data = torch.rand((1, num_inputs))  # Example random input
    spike_data = spikegen.rate(data, num_steps=50)  # Convert to spike trains
    loss_fn = SF.ce_rate_loss()  # Cross-entropy loss for spike-based training
    optimizer = torch.optim.Adam(snn_model.parameters(), lr=1e-3)
    num_epochs = 50
    num_steps = 50  # Number of time steps
    target = torch.randint(0, num_outputs, (1,))  # Random target for example

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        spk_rec, mem_rec = snn_model(spike_data)

        # Compute loss
        loss = loss_fn(mem_rec, target)

        # Backpropagation
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


if __name__ == '__main__':
    train()
