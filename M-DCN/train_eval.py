import datetime
import logging
import os
import time

import numpy as np
import torch

from load_data import Data
from mdcn import MDCN


class RunModel:

    def __init__(self, data: Data, modelname, optimizer_method="Adam", learning_rate=0.001, ent_vec_dim=200,
                 rel_vec_dim=200, num_iterations=100, batch_size=128, decay_rate=0., cuda=False, input_dropout=0.,
                 hidden_dropout=0., feature_map_dropout=0., in_channels=1, out_channels=32, filt_h=3, filt_w=3,
                 label_smoothing=0., num_to_eval=10, get_best_results=True, get_complex_results=True):

        self.cuda = cuda
        self.data = data
        self.model_name = modelname
        self.optimizer_method = optimizer_method
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.num_to_eval = num_to_eval
        self.get_best_results = get_best_results
        self.get_complex_results = get_complex_results
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout": hidden_dropout,
                       "feature_map_dropout": feature_map_dropout, "in_channels": in_channels,
                       "out_channels": out_channels, "filt_height": filt_h, "filt_width": filt_w}
        self.best_hits_10, self.best_hits_3, self.best_hits_1, self.best_mr, self.best_mrr = 0, 0, 0, 1e7, 0
        if self.get_complex_results:
            self.get_best_results = False

        now_time = datetime.datetime.now().strftime('_%m-%d-%H-%M-%S')
        logging.basicConfig(filename=os.path.join(os.getcwd(), 'results/%s.txt' % (
                self.model_name + "_" + self.data.data_name + str(now_time))),
                            level=logging.NOTSET, format='%(message)s')
        logging.info("data name: " + self.data.data_name)
        logging.info("model name: " + self.model_name)
        logging.info("learning rate: " + str(self.learning_rate))
        logging.info("entity dimension: " + str(self.ent_vec_dim))
        logging.info("relation dimension: " + str(self.rel_vec_dim))
        logging.info("batch size: " + str(self.batch_size))
        logging.info("decay rate: " + str(self.decay_rate))
        logging.info("label smoothing: " + str(self.label_smoothing))
        logging.info("convolution parameters: " + str(self.kwargs) + '\n')

        logging.info('Epoch' + '\t' + 'MR' + '\t' + 'MRR' + '\t' + 'Hits1' + '\t' + 'Hits3' + '\t' + 'Hits10' + '\t' +
                     'Best_MR' + '\t' + 'Best_MRR' + '\t' + 'Best_Hits1' + '\t' + 'Best_Hits3' + '\t' + 'Best_Hits10')

    def evaluate(self, model, eval_data, iteration):
        rank_filt = []
        hits_filt = []
        for i in range(10):
            hits_filt.append([])

        for batch_data, batch_num in self.data.get_batch_eval_data(self.batch_size, eval_data):
            head_id = batch_data[:, 0]
            rela_id = batch_data[:, 1]
            tail_id = batch_data[:, 2]
            if self.cuda:
                head_id = head_id.cuda()
                rela_id = rela_id.cuda()
                tail_id = tail_id.cuda()
            pred = model.forward(head_id, rela_id)

            # get filter rank
            for i in range(batch_num):
                filt = self.data.all_hr_dict[(head_id[i].item(), rela_id[i].item())]
                target_value = pred[i, tail_id[i]].item()
                pred[i, filt] = 0.0
                pred[i, tail_id[i]] = target_value

            _, filt_sort_id = torch.topk(pred, k=self.data.entities_num)
            filt_sort_id = filt_sort_id.cpu().numpy()

            for i in range(batch_num):
                rank_f = np.where(filt_sort_id[i] == tail_id[i].item())[0][0]
                rank_filt.append(rank_f + 1)

                for hits_level in range(10):
                    if rank_f <= hits_level:
                        hits_filt[hits_level].append(1.0)
                    else:
                        hits_filt[hits_level].append(0.0)

        filt_hits_10, filt_hits_3, filt_hits_1 = np.mean(hits_filt[9]), np.mean(hits_filt[2]), np.mean(hits_filt[0])
        filt_mr, filt_mrr = np.mean(rank_filt), np.mean(1. / np.array(rank_filt))

        # return the best results.
        if self.best_hits_10 < filt_hits_10:
            self.best_hits_10 = filt_hits_10
        if self.best_hits_3 < filt_hits_3:
            self.best_hits_3 = filt_hits_3
        if self.best_hits_1 < filt_hits_1:
            self.best_hits_1 = filt_hits_1
        if self.best_mr > filt_mr:
            self.best_mr = filt_mr
        if self.best_mrr < filt_mrr:
            self.best_mrr = filt_mrr

        if self.get_best_results:
            print('----- [%s]: [%s] results -----' % (self.model_name, self.data.data_name))
            print('Hits  @10: %.3f, Best  Hits @10: %.3f' % (filt_hits_10, self.best_hits_10))
            print('Hits   @3: %.3f, Best  Hits  @3: %.3f' % (filt_hits_3, self.best_hits_3))
            print('Hits   @1: %.3f, Best  Hits  @1: %.3f' % (filt_hits_1, self.best_hits_1))
            print('MR : %.3f, Best MR : %.3f' % (filt_mr, self.best_mr))
            print('MRR: %.3f, Best MRR: %.3f' % (filt_mrr, self.best_mrr))

            logging.info(str(iteration + 1) + '\t' + str('%.1f' % (filt_mr)) + '\t' + str('%.3f' % (filt_mrr)) + '\t' +
                         str('%.3f' % (filt_hits_1)) + '\t' + str('%.3f' % (filt_hits_3)) + '\t' +
                         str('%.3f' % (filt_hits_10)) + '\t' +
                         str('%.1f' % (self.best_mr)) + '\t' + str('%.3f' % (self.best_mrr)) + '\t' +
                         str('%.3f' % (self.best_hits_1)) + '\t' + str('%.3f' % (self.best_hits_3)) + '\t' +
                         str('%.3f' % (self.best_hits_10)))

        return filt_hits_10, filt_hits_3, filt_hits_1, filt_mr, filt_mrr

    def train_and_eval(self):
        if self.model_name.lower() == "mdcn":
            model = MDCN(self.data, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)

        # print([param.numel() for param in model.parameters()])
        # print([(name, param) for name, param in model.named_parameters()])

        if self.cuda:
            model.cuda()
        model.init()
        if self.optimizer_method.lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        elif self.optimizer_method.lower() == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.decay_rate)

        print('----- Start training -----')
        for iteration in range(self.num_iterations):
            begin_time = time.time()
            model.train()
            epoch_loss = 0
            for batch_data, batch_target in self.data.get_batch_train_data(self.batch_size):
                # Clears the gradients of all optimized
                optimizer.zero_grad()

                head_id = batch_data[:, 0]
                rela_id = batch_data[:, 1]
                if self.cuda:
                    head_id = head_id.cuda()
                    rela_id = rela_id.cuda()
                    batch_target = batch_target.cuda()

                pred = model.forward(head_id, rela_id)

                if self.label_smoothing:
                    batch_target = ((1.0 - self.label_smoothing) * batch_target) + (1.0 / batch_target.size(1))
                batch_loss = model.loss(pred, batch_target)

                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()

            if self.decay_rate:
                scheduler.step(epoch_loss)
            end_time = time.time()
            print('Iteration:' + str(iteration) + '   ' + 'epoch loss: {:.5f}'.format(epoch_loss) +
                  '   ' + 'time cost: {:.3f}'.format(end_time - begin_time))

            # logging.info('Iteration:' + str(iteration) + '   ' + 'mean loss: {:.5f}'.format(epoch_loss) +
            #              '   ' + 'time cost: {:.3f}'.format(time.time() - begin_time))

            model.eval()
            with torch.no_grad():
                if (iteration + 1) % self.num_to_eval == 0:
                    if self.get_complex_results:
                        pass
                    else:
                        # print("----- Valid_data evaluation-----")
                        # self.evaluate(model, data.valid_data_id)
                        print('----- Test_data evaluation -----')
                        self.evaluate(model, self.data.test_data_id, iteration)
