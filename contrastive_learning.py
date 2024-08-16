import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, similarity_scores):
        batch_size = similarity_scores.size(0)

        # Apply temperature scaling
        similarity_scores /= self.temperature

        # Split similarity scores into video-to-text and text-to-video scores
        similarity_vt = similarity_scores
        similarity_tv = similarity_scores.t()

        # Construct labels and calculate loss
        labels = torch.arange(batch_size).to(similarity_scores.device)
        loss_vt = self.loss(similarity_vt, labels)
        loss_tv = self.loss(similarity_tv, labels)
        return (loss_vt + loss_tv)/2


def cosine_sim(x, y):
    """
    cosine similarity between all the image and sentence pairs
    """
    inner_prod = x.mm(y.t())
    im_norm = torch.sqrt((x ** 2).sum(1).view(-1, 1) + 1e-18)
    s_norm = torch.sqrt((y ** 2).sum(1).view(1, -1) + 1e-18)
    sim = inner_prod / (im_norm * s_norm)
    return sim
