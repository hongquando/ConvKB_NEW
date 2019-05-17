import multiprocessing
import time
import torch
from torch.autograd import Variable
import numpy as np
from scipy.stats import rankdata
import random
import math

class MyProcessConvKB(multiprocessing.Process):
    def __init__(self, L, net, tripleDict,candidates, queue=None):
        super(MyProcessConvKB, self).__init__()
        self.L = L
        self.net = net
        self.queue = queue
        self.tripleDict = tripleDict
        self.candidates = candidates

    def run(self):
        while True:
            testList = self.queue.get()
            try:
                self.process_data(testList,self.net, self.tripleDict,
                                  self.candidates, self.L)
            except:
                time.sleep(5)
                self.process_data(testList,self.net, self.tripleDict,
                              self.candidates, self.L)
            self.queue.task_done()

    def process_data(self, testList, net ,tripleDict,candidates, L):
        hit10Count, totalRank, mrr,tripleCount = evaluation_ConvKB_helper(testList,net, tripleDict,candidates)
        L.append((hit10Count, totalRank,mrr, tripleCount))

def evaluation_ConvKB_helper(testList,net, tripleDict,candidates):
    # Evaluate the prediction of only tail entities
    #print("Evaluate the prediction of only tail entities")
    hits10 = 0.0
    mr = 0.0
    mrr = 0.0
    new_candidates = set(random.choices(candidates, k=128))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for triple in testList:
        new_candidates.add(triple.t)
    for triple in testList:
        h_batch = []
        t_batch = []
        r_batch = []
        h_batch.append(triple.h)
        t_batch.append(triple.t)
        r_batch.append(triple.r)
        #print("1",triple.h,triple.t,triple.r)
        for att in new_candidates:
            if (triple.h, att, triple.r) in tripleDict:
                continue
            h_batch.append(triple.h)
            t_batch.append(att)
            r_batch.append(triple.r)
            # print("2",triple.h, triple.t, triple.r)
        if torch.cuda.is_available():
            h_batch, t_batch, r_batch = h_batch.to(device), t_batch.to(device), r_batch.to(device)
        h_batch, t_batch, r_batch = Variable(h_batch), Variable(t_batch), Variable(r_batch)
        outputs, _, _, _ = net(h_batch, t_batch, r_batch)
        outputs = 1 - outputs.view(-1) / torch.max(torch.abs(outputs))
        outputs = outputs.data.tolist()
        results_with_id = rankdata(outputs, method='ordinal')
        _filter = results_with_id[0]
        mr += _filter
        mrr += 1.0/_filter
        if _filter <= 10:
            hits10 += 1
        # print('Meanrank: %.6f' % totalRank)
        # print('Hit@10: %.6f' % hit10Count)
        # print('MRR: %.6f' % mrr)
    tripleCount = len(testList)
    return hits10, mr, mrr, tripleCount

def evaluation_ConvKB(testList, net, tripleDict, candidates, k=0, num_processes=multiprocessing.cpu_count()):
    if k > len(testList):
        testList = random.choices(testList, k=k)
    elif k > 0:
        testList = random.sample(testList, k=k)
    # Split the testList into #num_processes parts
    len_split = math.ceil(len(testList) / num_processes)
    testListSplit = [testList[i: i + len_split] for i in range(0, len(testList), len_split)]
    with multiprocessing.Manager() as manager:
        # Create a public writable list to store the result
        print("Create a public writable list to store the result")
        L = manager.list()
        queue = multiprocessing.JoinableQueue()
        workerList = []
        for i in range(num_processes):
            worker = MyProcessConvKB(L, net, tripleDict, candidates,queue=queue)
            workerList.append(worker)
            worker.daemon = True
            worker.start()

        for subList in testListSplit:
            queue.put(subList)

        queue.join()

        resultList = list(L)

        # Terminate the worker after execution, to avoid memory leaking
        for worker in workerList:
            worker.terminate()

    hits10 = sum([elem[0] for elem in resultList]) / len(testList)
    meanrank = sum([elem[1] for elem in resultList])
    mrr = sum([elem[2] for elem in resultList]) / len(testList)
    print('Meanrank: %.6f' % meanrank)
    print('Hit@10: %.6f' % hits10)
    print('Mrr@10: %.6f' % mrr)
    return hits10, meanrank, mrr

