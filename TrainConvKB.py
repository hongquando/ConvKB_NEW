from sklearn.neighbors import NearestNeighbors
from model.TransE import *
from model.ConvKB import ConvKB
from model.utils import *
from model.CustomTripletMarginLoss import CustomTripletMarginLoss
from argparse import Namespace
import torch
import torch.nn as nn
import sys
from torch.autograd import Variable
import torch.optim as optim
from numpy import linalg as LA
import numpy as np
from math import log10,floor
import json
import pickle

args = Namespace(
    entity_path='./data/GENE/entity2id.txt',

    relation_path='./data/GENE/relation2id.txt',

    triplets_path='./data/GENE/triplet2id.txt',
    embedding_size=100,
    batch_size=128,

    seed=0,
    log_interval=15,
    display_step=5,

    trans_e_margin=1,
    trans_e_weight_decay=0.001,
    trans_e_learning_rate=5e-4,
    trans_e_n_epochs=1,
    trans_e_save_path='./data/GENE/TransE.pkl',

    conv_kb_weight_decay=0.001,
    conv_kb_learning_rate=1e-4,
    conv_kb_n_epochs=1,
    conv_kb_momentum=0.9,
    new_conv_kb_save_path='./data/GENE/TempConvKB.pkl',
    conv_kb_save_path='./data/GENE/ConvKB.pkl'
)
class TrainConvKB():
    net = None
    # processed_entity_2_id = dict()
    # relation_2_id = dict()
    # triplets = dict()
    def __init__(self):
        #super(TrainConvKB,self).__init__(0)
        self.entity_total = get_total(args.entity_path)
        self.relation_total = get_total(args.relation_path)
        if os.path.exists(args.entity_path):
            self.processed_entity_2_id = load_data(args.entity_path, ignore_first=True)

        if os.path.exists(args.relation_path):
            self.relation_2_id = load_data(args.relation_path, ignore_first=True)

        if os.path.exists(args.triplets_path):
            self.triplets = load_data(args.triplets_path, is_triplet=True, ignore_first=True)

        if os.path.exists(args.conv_kb_save_path) and os.path.exists(args.entity_path) and os.path.exists(
                args.relation_path):
            self.net = ConvKB(self.entity_total, self.relation_total, args.embedding_size)
            if torch.cuda.is_available():
                self.net = self.net.cuda()
                self.net.load_state_dict(torch.load(args.conv_kb_save_path))
            else:
                self.net.load_state_dict(torch.load(args.conv_kb_save_path, map_location=lambda storage, loc: storage))
            self.train()
        #self.net.eval()

    def cleanup(self):
        self.persist()

    def persist(self):
        print('Saving model...')
        with open(args.entity_path, 'w') as f:
            f.write('{}\n'.format(len(self.processed_entity_2_id)))
            for processed_entity, idx in sorted(list(self.processed_entity_2_id.items()),
                                                key=lambda kv: (kv[1])):
                f.write('{}\t{}\n'.format(processed_entity, idx))

        with open(args.relation_path, 'w') as f:
            f.write('{}\n'.format(len(self.relation_2_id)))
            for relation, idx in sorted(list(self.relation_2_id.items()), key=lambda kv: int(kv[1])):
                f.write('{}\t{}\n'.format(relation, idx))

        print('Saved model to file')

    def get_item_embedding(self, item_id):
        key = "_item:" + str(item_id)
        if key in self.processed_entity_2_id:
            idx = self.processed_entity_2_id[key]
            idx = torch.LongTensor([idx])
            if torch.cuda.is_available():
                idx = idx.cuda()
            idx = Variable(idx)
            embedding = self.net.ent_embeddings(idx).data[0].cpu().numpy()
            norm = LA.norm(embedding)
            if norm == 0:
                return embedding
            return embedding / LA.norm(embedding)
        return None

    def train_TransE(self,entity_total,relation_total,triplets,n_epochs=None):
        net = TransE(entity_total,relation_total,args.embedding_size)
        if self.net is not None:
            embedding_entities = np.random.normal(0, 0.01, (entity_total, args.embedding_size))
            embedding_entities[:self.entity_total] = self.net.ent_embeddings.weight.data.cpu().numpy()
            net.ent_embeddings.weight.data.copy_(torch.from_numpy(embedding_entities))

            embedding_relations = np.random.normal(0, 0.01, (relation_total, args.embedding_size))
            embedding_relations[:self.relation_total] = self.net.rel_embeddings.weight.data.cpu().numpy()
            net.rel_embeddings.weight.data.copy_(torch.from_numpy(embedding_relations))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        print("Using CUDA: {}".format(next(net.parameters()).is_cuda))
        net.train()
        optimizer = optim.Adam(net.parameters(), lr=args.trans_e_learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, min_lr=1e-5,
                                                         verbose=True)
        criterion = CustomTripletMarginLoss(margin=args.trans_e_margin)

        # 2. Load triples #
        triple_total, triple_list, triple_dict, tails_per_head, heads_per_tail = load_triplet_2(triplets)

        # 4. Train #
        min_loss = None
        if n_epochs is None:
            n_epochs = args.trans_e_n_epochs

        for epoch in range(1, n_epochs + 1):  # loop over the dataset multiple times
            # shuffle train set
            random.shuffle(triple_list)
            acc_loss = 0.0

            n_batches = triple_total // args.batch_size
            if (triple_total - n_batches * args.batch_size) != 0:
                n_batches += 1
            for batch_idx, i in enumerate(range(n_batches), 1):
                start = i * args.batch_size
                end = min([start + args.batch_size, triple_total])

                triple_batch = triple_list[start:end]
                pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch = get_batch_filter_all(
                    triple_batch, entity_total, triple_dict, tails_per_head, heads_per_tail)

                pos_h_batch, neg_h_batch = torch.LongTensor(pos_h_batch), torch.LongTensor(neg_h_batch)
                pos_t_batch, neg_t_batch = torch.LongTensor(pos_t_batch), torch.LongTensor(neg_t_batch)
                pos_r_batch, neg_r_batch = torch.LongTensor(pos_r_batch), torch.LongTensor(neg_r_batch)

                pos_h_batch, neg_h_batch = pos_h_batch.to(device), neg_h_batch.to(device)
                pos_t_batch, neg_t_batch = pos_t_batch.to(device), neg_t_batch.to(device)
                pos_r_batch, neg_r_batch = pos_r_batch.to(device), neg_r_batch.to(device)

                pos_h_batch, neg_h_batch = Variable(pos_h_batch), Variable(neg_h_batch)
                pos_t_batch, neg_t_batch = Variable(pos_t_batch), Variable(neg_t_batch)
                pos_r_batch, neg_r_batch = Variable(pos_r_batch), Variable(neg_r_batch)

                # zero the parameter gradients
                optimizer.zero_grad()
                pos, neg, pos_h_e, pos_t_e, neg_h_e, neg_t_e = net(pos_h_batch, pos_t_batch, pos_r_batch,
                                                                   neg_h_batch, neg_t_batch, neg_r_batch)

                ent_embeddings = net.ent_embeddings(torch.cat([pos_h_batch, pos_t_batch, neg_h_batch, neg_t_batch]))
                rel_embeddings = net.rel_embeddings(torch.cat([pos_r_batch, neg_r_batch]))

                loss_triplet = criterion(pos, neg)
                norm_loss = ent_embeddings.norm(2) + rel_embeddings.norm(2)
                norm_loss += pos_h_e.norm(2) + pos_t_e.norm(2) + neg_h_e.norm(2) + neg_t_e.norm(2)

                loss = loss_triplet + args.trans_e_weight_decay * norm_loss
                batch_loss = loss.item()
                loss.backward()
                optimizer.step()

                acc_loss += batch_loss

                if batch_idx % args.log_interval == 0:
                    offset = int(floor(log10(n_batches)) - floor(log10(batch_idx)))
                    print('\r\033[K\rTrain Epoch: {} [{}{} / {} ({:.0f}%)]   Learning Rate: {}   Loss: {:.6f}'
                          .format(epoch, batch_idx, ' ' * offset, n_batches, 100. * batch_idx / n_batches,_get_learning_rate(optimizer)[0], batch_loss)),
                    sys.stdout.flush()

            acc_loss /= n_batches
            # print statistics
            if epoch % args.display_step == 0 or epoch == 1:
                print('\r\033[K\r[{:3d}] loss: {:.5f} - learning rate: {}'
                      .format(epoch, acc_loss, _get_learning_rate(optimizer)[0]))

            if min_loss is None or acc_loss < min_loss:
                min_loss = acc_loss
                with open(args.trans_e_save_path, 'wb') as f:
                    torch.save(net.state_dict(), f)
            scheduler.step(acc_loss, epoch)

        print('\nFinished Training\n')

        if torch.cuda.is_available():
            net.load_state_dict(torch.load(args.trans_e_save_path))
        else:
            net.load_state_dict(torch.load(args.trans_e_save_path, map_location=lambda storage, loc: storage))
        return net

    def train_ConvKB(self, ent_embeddings, rel_embeddings, triplets, n_epochs=None):
        # 1. Initial net, criterion, optimizer and scheduler (if needed) #
        entity_total = ent_embeddings.shape[0]
        relation_total = rel_embeddings.shape[0]

        net = ConvKB(entity_total, relation_total, args.embedding_size)
        net.set_pretrained_weights(ent_embeddings, rel_embeddings)
        # net.ent_embeddings.weight.requires_grad = False
        # net.rel_embeddings.weight.requires_grad = False

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        print("Using CUDA: {}".format(next(net.parameters()).is_cuda))

        net.train()
        params_dict = dict(net.named_parameters())
        net_params = []
        for key, value in params_dict.items():
            if not value.requires_grad:
                continue
            if key.startswith('fc'):
                net_params += [{'params': [value], 'weight_decay': args.conv_kb_weight_decay}]
            else:
                net_params += [{'params': [value]}]
        optimizer = optim.Adam(net_params, lr=args.conv_kb_learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, min_lr=1e-5,
                                                         verbose=True)
        criterion = nn.SoftMarginLoss()

        # 2. Load triples #
        triple_total, triple_list, triple_dict, tails_per_head, heads_per_tail = load_triplet_2(triplets)

        # 3. Train #
        min_loss = None
        if n_epochs is None:
            n_epochs = args.conv_kb_n_epochs

        for epoch in range(1, n_epochs + 1):  # loop over the dataset multiple times
            # shuffle train set
            random.shuffle(triple_list)
            acc_loss = 0.0

            n_batches = triple_total // args.batch_size
            if (triple_total - n_batches * args.batch_size) != 0:
                n_batches += 1
            for batch_idx, i in enumerate(range(n_batches), 1):
                start = i * args.batch_size
                end = min([start + args.batch_size, triple_total])

                triple_batch = triple_list[start:end]
                pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch = get_batch_filter_all(
                    triple_batch, entity_total, triple_dict, tails_per_head, heads_per_tail)

                h_batch, t_batch, r_batch, targets = [], [], [], []
                for h, t, r in zip(pos_h_batch, pos_t_batch, pos_r_batch):
                    h_batch.append(h)
                    t_batch.append(t)
                    r_batch.append(r)
                    targets.append([-1.])

                for h, t, r in zip(neg_h_batch, neg_t_batch, neg_r_batch):
                    h_batch.append(h)
                    t_batch.append(t)
                    r_batch.append(r)
                    targets.append([1.])

                h_batch, t_batch = torch.LongTensor(h_batch), torch.LongTensor(t_batch)
                r_batch, targets = torch.LongTensor(r_batch), torch.FloatTensor(targets)

                h_batch, t_batch = h_batch.to(device), t_batch.to(device)
                r_batch, targets = r_batch.to(device), targets.to(device)

                h_batch, t_batch = Variable(h_batch), Variable(t_batch)
                r_batch, targets = Variable(r_batch), Variable(targets)

                # zero the parameter gradients
                optimizer.zero_grad()
                outputs, h_e, t_e, r_e = net(h_batch, t_batch, r_batch)

                # ent_embeddings = net.ent_embeddings(torch.cat([pos_h_batch, pos_t_batch, neg_h_batch, neg_t_batch]))
                # rel_embeddings = net.rel_embeddings(torch.cat([pos_r_batch, neg_r_batch]))

                loss_triplet = criterion(outputs, targets)
                norm_loss = h_e.norm(2) + t_e.norm(2) + r_e.norm(2)

                loss = loss_triplet + args.conv_kb_weight_decay * norm_loss
                batch_loss = loss.item()
                loss.backward()
                optimizer.step()

                acc_loss += batch_loss

                if batch_idx % args.log_interval == 0:
                    offset = int(floor(log10(n_batches)) - floor(log10(batch_idx)))
                    print('\r\033[K\rTrain Epoch: {} [{}{} / {} ({:.0f}%)]   Learning Rate: {}   Loss: {:.6f}'
                          .format(epoch, batch_idx, ' ' * offset, n_batches, 100. * batch_idx / n_batches,
                                  _get_learning_rate(optimizer)[0], batch_loss)),
                    sys.stdout.flush()

            acc_loss /= n_batches
            # print statistics
            if epoch % args.display_step == 0 or epoch == 1:
                print('\r\033[K\r[{:3d}] loss: {:.5f} - learning rate: {}'
                      .format(epoch, acc_loss, _get_learning_rate(optimizer)[0]))

            if min_loss is None or acc_loss < min_loss:
                min_loss = acc_loss
                with open(args.conv_kb_save_path, 'wb') as f:
                    torch.save(net.state_dict(), f)
            scheduler.step(acc_loss, epoch)
        print('\nFinished Training\n')

        if torch.cuda.is_available():
            net.load_state_dict(torch.load(args.conv_kb_save_path))
        else:
            net.load_state_dict(torch.load(args.conv_kb_save_path, map_location=lambda storage, loc: storage))
        return net

    def get_item_embedding(self, item_id):
        key = "_item:" + str(item_id)
        if key in self.processed_entity_2_id:
            idx = self.processed_entity_2_id[key]
            idx = torch.LongTensor([idx])
            if torch.cuda.is_available():
                idx = idx.cuda()
            idx = Variable(idx)
            embedding = self.net.ent_embeddings(idx).data[0].cpu().numpy()
            norm = LA.norm(embedding)
            if norm == 0:
                return embedding
            return embedding / LA.norm(embedding)
        return None

    def train(self,trans_e_n_epochs=None, conv_kb_n_epochs=None):
        net = self.train_TransE(self.entity_total, self.relation_total, self.triplets, n_epochs=trans_e_n_epochs)
        ent_embeddings = net.ent_embeddings.weight.data.cpu().numpy()
        rel_embeddings = net.rel_embeddings.weight.data.cpu().numpy()
        self.net = net
        # self.processed_entity_2_id.update(processed_entity_2_id)
        # self.relation_2_id.update(relation_2_id)
        # self.triplets.update(triplets)
        net = self.train_ConvKB(ent_embeddings, rel_embeddings, self.triplets, n_epochs=conv_kb_n_epochs)

def _get_learning_rate(o):
    lr = []
    for param_group in o.param_groups:
        lr += [param_group['lr']]
    return lr

if __name__ == '__main__':
    if not os.path.exists(args.conv_kb_save_path) and not os.path.exists(args.trans_e_save_path):
        TrainConvKB().train()
    if torch.cuda.is_available():
        net = torch.load(args.conv_kb_save_path)
    else:
        net = torch.load(args.conv_kb_save_path, map_location=lambda storage, loc: storage)
    net = list(net.items())
    # 1: entity
    # 2: relation
    data_train = net[0][1].cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=15, algorithm='ball_tree').fit(data_train)
    distances, indices = nbrs.kneighbors(data_train)
    with open("./data/GENE/kNN.pkl", "wb") as f:
        pickle.dump(nbrs,f)
        f.close()
    with open("./data/GENE/indices.pkl", "wb") as f:
        pickle.dump(indices, f)
        f.close()
    with open("./data/GENE/distance.pkl", "wb") as f:
        pickle.dump(distances, f)
        f.close()
    # with open("./data/GENE/db.json", "rb") as f:
    #     data = json.load(f)
    #     f.close()
    # with open("./data/GENE/id_dict", "rb") as f:
    #     id_dict = pickle.load(f)
    #     f.close()
    # processed_entity_2_id = load_data(args.entity_path, ignore_first=True)
    # relation_2_id = load_data(args.relation_path, ignore_first=True)
    # processed_id_2_entity = dict()
    # i = 0
    # with open(args.entity_path, 'r') as f:
    #     for line in f:
    #         if True and i == 0:
    #             i += 1
    #             continue
    #         line = line.strip()
    #         if line == '':
    #             continue
    #         parts = line.split("\t")
    #         processed_id_2_entity[int(parts[1])] = parts[0]
    # while True:
    #     gene_name = input("\nType genename: ").strip()
    #     if gene_name == "":
    #         break
    #     if gene_name not in processed_entity_2_id.keys():
    #         print("Gene not found")
    #         continue
    #     print("Gene {}\n{}".format(gene_name, data[id_dict[gene_name]]))
    #     print("Top 5 gene related: ")
    #     count = 0
        # for index in indices[processed_entity_2_id[gene_name]][1:]:
        #     similar_gene = processed_id_2_entity[index]
        #     if similar_gene in id_dict.keys():
        #         if id_dict[similar_gene] in data.keys():
        #             print("Gene {}\n{}".format(
        #                 similar_gene, data[id_dict[similar_gene]]))
        #             count+=1
        #     if count == 5: break
            # else:
            #     print("Att {}\n".format(similar_gene))
            #     count += 1
            #     if count == 5: break


