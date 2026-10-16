import torch
import torch.nn as nn
import torch.nn.functional as F
from utscls.attncls import AttnClassification


class Model(nn.Module):

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.vm = AttnClassification(configs)

    def forecast(self, x_enc):
        raise NotImplementedError()

    def imputation(self, x_enc):
        raise NotImplementedError()

    def anomaly_detection(self, x_enc):
        raise NotImplementedError()

    def classification(self, x_enc):
        return self.vm.forward(x_enc)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
