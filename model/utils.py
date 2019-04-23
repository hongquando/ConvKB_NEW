from model.Triplet import Triplet
from copy import deepcopy
import random
import os

def load_data(file_path, is_triplet=False, ignore_first=False):
    i = 0
    result = dict()
    with open(file_path, 'r') as f:
        for line in f:
            if ignore_first and i == 0:
                i += 1
                continue
            line = line.strip()
            if line == '':
                continue
            parts = line.split("\t")
            if is_triplet:
                head = int(parts[0])
                tail = int(parts[1])
                rel = int(parts[2])
                result[(head, tail)] = rel
                continue
            result[parts[0]] = int(parts[1])
    return result

def load_triplet_2(triplets):
    bernoulli_dict = dict()
    triple_list = []
    triple_total = 0
    for (head, tail), rel in list(triplets.items()):
        heads, tails = bernoulli_dict.get(rel, (dict(), dict()))
        tail_per_head = heads.get(head, set())
        tail_per_head.add(tail)
        heads[head] = tail_per_head

        head_per_tail = tails.get(tail, set())
        head_per_tail.add(head)
        tails[tail] = head_per_tail

        bernoulli_dict[rel] = (heads, tails)
        triple_total += 1
        triple_list.append(Triplet(head, tail, rel))

    triple_dict = {}
    for triple in triple_list:
        triple_dict[(triple.h, triple.t, triple.r)] = True

    tails_per_head = {rel: sum(len(tail_per_head) for tail_per_head in heads.values()) / len(heads) for
                      rel, (heads, tails) in bernoulli_dict.items()}
    heads_per_tail = {rel: sum(len(head_per_tail) for head_per_tail in tails.values()) / len(tails) for
                      rel, (heads, tails) in bernoulli_dict.items()}
    return triple_total, triple_list, triple_dict, tails_per_head, heads_per_tail

# Change the head of a triple randomly,
# with checking whether it is a false negative sample.
# If it is, regenerate.
def corrupt_head_filter(triple, entity_total, triple_dict):
    new_triple = deepcopy(triple)
    while True:
        new_head = random.randrange(entity_total)
        if (new_head, new_triple.t, new_triple.r) not in triple_dict:
            break
    new_triple.h = new_head
    return new_triple


# Change the tail of a triple randomly,
# with checking whether it is a false negative sample.
# If it is, regenerate.
def corrupt_tail_filter(triple, entity_total, triple_dict):
    new_triple = deepcopy(triple)
    while True:
        new_tail = random.randrange(entity_total)
        if (new_triple.h, new_tail, new_triple.r) not in triple_dict:
            break
    new_triple.t = new_tail
    return new_triple


# Corrupt the head or tail according to Bernoulli Distribution,
# with checking whether it is a false negative sample.
def corrupt_filter_two_v2(triple, entity_total, triple_dict, tail_per_head, head_per_tail):
    rel = triple.r
    split = tail_per_head[rel] / (tail_per_head[rel] + head_per_tail[rel])
    random_number = random.random()
    if random_number < split:
        new_triple = corrupt_head_filter(triple, entity_total, triple_dict)
    else:
        new_triple = corrupt_tail_filter(triple, entity_total, triple_dict)
    return new_triple


# Use all the tripleList,
# and generate negative samples by corrupting head or tail with equal probabilities,
# with checking whether false negative samples exist.
def get_batch_filter_all(triple_list, entity_total, triple_dict, tail_per_head, head_per_tail):
    new_triple_list = [corrupt_filter_two_v2(triple, entity_total, triple_dict, tail_per_head, head_per_tail) for triple in triple_list]
    ph, pt, pr = get_three_elements(triple_list)
    nh, nt, nr = get_three_elements(new_triple_list)
    return ph, pt, pr, nh, nt, nr


def get_three_elements(triple_list):
    head_list = [triple.h for triple in triple_list]
    tail_list = [triple.t for triple in triple_list]
    rel_list = [triple.r for triple in triple_list]
    return head_list, tail_list, rel_list

def get_total(file_name):
    if not os.path.exists(file_name):
        return 0
    with open(file_name, 'r') as fr:
        for line in fr:
            return int(line)

if __name__ == '__main__':
    #r = load_data("/Users/mac/PycharmProjects/ConvKB/data/WN11/relation2id.txt",ignore_first=True)
    #h = load_data("/Users/mac/PycharmProjects/ConvKB/data/WN11/entity2id.txt", ignore_first=True)
    triplets = load_data("/Users/mac/PycharmProjects/ConvKB/data/WN11/triple2id.txt",is_triplet=True,ignore_first=True)
    triple_total, triple_list, triple_dict, tails_per_head, heads_per_tail = load_triplet_2(triplets)
    # h,t,r = get_three_elements(triple_list)
    # print("head:",len(h))
    # print("tail:",len(t))
    # print("relation:",len(r))
    print(tails_per_head)
    print(heads_per_tail)
    entity_total = get_total("/Users/mac/PycharmProjects/ConvKB/data/WN11/entity2id.txt")
    start = 0
    end = 200
    triple_batch = triple_list[start:end]
    h, t, r = get_three_elements(triple_list)
    #print("head:",h)
    print("relation:", r)
    pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch = get_batch_filter_all(
        triple_batch, entity_total, triple_dict, tails_per_head, heads_per_tail)