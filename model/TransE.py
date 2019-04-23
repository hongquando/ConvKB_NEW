import torch
import torch.nn as nn
import torch.nn.functional as F

class TransE(nn.Module):
    def __init__(self, entity_total, relation_total, embedding_size):
        super(TransE, self).__init__()
        self.embedding_size = embedding_size
        self.entity_total = entity_total
        self.relation_total = relation_total

        ent_weight = torch.FloatTensor(self.entity_total, self.embedding_size)
        rel_weight = torch.FloatTensor(self.relation_total, self.embedding_size)
        # Use xavier initialization method to initialize embeddings of entities and relations
        nn.init.xavier_uniform_(ent_weight)
        nn.init.xavier_uniform_(rel_weight)
        self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)

        normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        self.ent_embeddings.weight.data = normalize_entity_emb
        self.rel_embeddings.weight.data = normalize_relation_emb

    def set_pretrained_weights(self, ent_embeddings, rel_embeddings):
        self.ent_embeddings.weight = nn.Parameter(torch.from_numpy(ent_embeddings).float())
        self.rel_embeddings.weight = nn.Parameter(torch.from_numpy(rel_embeddings).float())

    def forward(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        pos_r_e = self.rel_embeddings(pos_r)
        neg_h_e = self.ent_embeddings(neg_h)
        neg_t_e = self.ent_embeddings(neg_t)
        neg_r_e = self.rel_embeddings(neg_r)

        pos = pos_h_e + pos_r_e - pos_t_e
        neg = neg_h_e + neg_r_e - neg_t_e
        return pos, neg, pos_h_e, pos_t_e, neg_h_e, neg_t_e