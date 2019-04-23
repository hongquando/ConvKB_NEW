import torch
import torch.nn as nn


class ConvKB(nn.Module):
    def __init__(self, entity_total, relation_total, embedding_size, num_filters=50):
        super(ConvKB, self).__init__()

        self.embedding_size = embedding_size
        self.num_filters = num_filters
        self.entity_total = entity_total
        self.relation_total = relation_total

        self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)

        self.conv = nn.Conv2d(1, num_filters, (3, 1))
        weights = torch.FloatTensor(num_filters * [0.1, 0.1, -0.1]).view(num_filters, 1, 3, 1)
        self.conv.weight.data = nn.Parameter(weights)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(embedding_size * num_filters, 1)

    def set_pretrained_weights(self, ent_embeddings, rel_embeddings):
        self.ent_embeddings.weight = nn.Parameter(torch.from_numpy(ent_embeddings).float())
        self.rel_embeddings.weight = nn.Parameter(torch.from_numpy(rel_embeddings).float())

    def forward(self, h, t, r):
        h_e = self.ent_embeddings(h).view(-1, 1, 1, self.embedding_size)
        t_e = self.ent_embeddings(t).view(-1, 1, 1, self.embedding_size)
        r_e = self.rel_embeddings(r).view(-1, 1, 1, self.embedding_size)

        x = torch.cat([h_e, r_e, t_e], 2)
        x = self.relu(self.conv(x))
        x = x.view(-1, self.embedding_size * self.num_filters)
        f = self.fc(x)

        return f, h_e, t_e, r_e
