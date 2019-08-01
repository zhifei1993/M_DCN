from collections import defaultdict

import numpy as np
import torch


class Data:

    def __init__(self, data_dir, reverse=False):
        self.data_name = data_dir[5:-1]
        self.train_data, self.train_data_num, self.train_hrt = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data, self.valid_data_num, self.valid_hrt = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data, self.test_data_num, self.test_hrt = self.load_data(data_dir, "test", reverse=reverse)
        self.all_data = self.train_data + self.valid_data + self.test_data
        self.all_hrt = self.train_hrt + self.valid_hrt + self.test_hrt

        self.entities, self.entities_id, self.entities_num = self.get_entities(self.all_data)
        self.relations, self.relations_id, self.relations_num = self.get_relations(self.all_data)

        self.train_data_id = self.data_id(self.train_data)
        self.valid_data_id = self.data_id(self.valid_data)
        self.test_data_id = self.data_id(self.test_data)
        self.all_data_id = self.data_id(self.all_data)

        self.train_hr_dict = self.get_hr_dict(self.train_data_id)
        self.train_hr_list = list(self.train_hr_dict.keys())
        self.train_hr_list_num = len(self.train_hr_list)
        self.all_hr_dict = self.get_hr_dict(self.all_data_id)

        print('datasets: {}'.format(self.data_name))
        print('enti_num: {}'.format(self.entities_num), 'rela_num: {}'.format(self.relations_num))
        print('train_num:{}'.format(self.train_data_num),
              'valid_num:{}'.format(self.valid_data_num),
              'test_num: {}'.format(self.test_data_num))

    @staticmethod
    def get_hr_dict(data_id):
        hr_dict = defaultdict(list)
        for triple in data_id:
            hr_dict[(triple[0], triple[1])].append(triple[2])
        return hr_dict

    def get_batch_train_data(self, batch_size):
        start = 0
        np.random.shuffle(self.train_hr_list)
        while start < self.train_hr_list_num:
            end = min(start + batch_size, self.train_hr_list_num)
            batch_data = self.train_hr_list[start:end]

            batch_target = np.zeros((len(batch_data), self.entities_num))
            for index, hr_pair in enumerate(batch_data):
                batch_target[index, self.train_hr_dict[hr_pair]] = 1.0

            batch_data = torch.tensor(batch_data)
            batch_target = torch.FloatTensor(batch_target)

            start = end
            yield batch_data, batch_target

    @staticmethod
    def get_batch_eval_data(batch_size, eval_data):
        eval_data_num = len(eval_data)
        start = 0
        while start < eval_data_num:
            end = min(start + batch_size, eval_data_num)

            batch_data = eval_data[start:end]
            batch_num = len(batch_data)
            batch_data = torch.tensor(batch_data)

            start = end
            yield batch_data, batch_num

    def data_id(self, data):
        data_num = len(data)
        data_id = [(self.entities_id[data[i][0]],
                    self.relations_id[data[i][1]],
                    self.entities_id[data[i][2]])
                   for i in range(data_num)]
        return data_id

    @staticmethod
    def load_data(data_dir, data_type, reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            hrt = f.read().strip().split("\n")
            hrt = [i.split() for i in hrt]
            trh = []
            if reverse:
                trh = [[i[2], i[1] + "_reverse", i[0]] for i in hrt]
            data = hrt + trh
            data_num = len(data)
            f.close()
        return data, data_num, hrt

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        entities_num = len(entities)
        entities_id = {entities[i]: i for i in range(entities_num)}
        return entities, entities_id, entities_num

    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        relations_num = len(relations)
        relations_id = {relations[i]: i for i in range(relations_num)}
        return relations, relations_id, relations_num


# Data(data_dir="data/FB15k/", reverse=True)
