import collections

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils import data as data

from bnn import BNet
from dataset import Dataset


class BasePipeline(object):
    """
    Class defining a training pipeline. Instantiates data loaders, model,
    and optimizer. Handles training for multiple epochs and keeping track of
    train and test loss.
    """

    def __init__(self,
                 train,
                 input_size,
                 output_size,
                 test=None,
                 model=BNet,
                 batch_size=32,
                 optimizer=torch.optim.Adam,
                 lr=1e-4,
                 loss_function=nn.MSELoss(reduction='sum'),
                 n_epochs=10,
                 verbose=False,
                 eval_metrics=None,
                 random_seed=None):
        self.train = train
        self.test = test

        self.train_loader = data.DataLoader(
            train, batch_size=batch_size, shuffle=True)
        if self.test is not None:
            self.test_loader = data.DataLoader(
                test, batch_size=batch_size, shuffle=True)
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.n_epochs = n_epochs

        self.model = model(
            input_size,
            batch_size,
            output_size,
            len(self.train),
            loss_function=loss_function)
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.losses = collections.defaultdict(list)
        self.verbose = verbose
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        if eval_metrics is None:
            eval_metrics = []
        self.eval_metrics = eval_metrics

    def fit(self):
        # set model to training mode
        for epoch in range(1, self.n_epochs + 1):
            self.model.train()
            train_loss = self._fit_epoch(epoch)
            self.losses['train'].append(train_loss)
            row = 'Epoch: {0:^3}  train: {1:^10.5f}'.format(
                epoch, self.losses['train'][-1])
            if self.test is not None:
                self.losses['test'].append(self._validation_loss())
                row += 'val: {0:^10.5f}'.format(self.losses['test'][-1])
                for metric in self.eval_metrics:
                    row += self._evaluate(metric, self.model, self.test_loader)
            self.losses['epoch'].append(epoch)
            if self.verbose:
                print(row)

    def _fit_epoch(self, epoch=1, queue=None):
        total_loss = torch.Tensor([0])
        for X, y in self.train_loader:
            self.optimizer.zero_grad()

            loss, _, _, _ = self.model.elbo(X, y)
            # preds = self.model(X)
            # loss = self.loss_function(preds, y)
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()
            batch_loss = loss.item() / y.size()[0]
            print('{} loss {}'.format(epoch, batch_loss))
        total_loss /= len(self.train)
        if queue is not None:
            queue.put(total_loss[0])
        else:
            return total_loss[0]

    def _validation_loss(self):
        self.model.eval()
        total_loss = torch.Tensor([0])
        for X, y in self.test_loader:
            preds = self.model(X)
            loss = self.loss_function(preds, y)
            total_loss += loss.item()

        total_loss /= len(self.test)
        return total_loss[0]

    def _evaluate(self, metric, model, dataloader, row=''):
        res = []
        for X, y in dataloader:
            pred = model(X)
            res.append(metric(pred, y))
        res = torch.mean(torch.from_numpy(np.asarray(res)))
        self.losses['eval-{}'.format(metric)].append(res)
        row += 'eval-{0}: {1:^10.5f}'.format(metric, res)
        return row
