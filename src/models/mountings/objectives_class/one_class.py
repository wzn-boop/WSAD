import torch


class OCLoss(torch.nn.Module):
    def __init__(self, c, nac=False, unc=False, alpha=0.1, reduction='mean'):
        super(OCLoss, self).__init__()
        self.c = c
        self.reduction = reduction

        self.nac = nac
        self.unc = unc
        self.alpha = alpha

        if unc is True:
            self.unc_svdd_loss = UncSVDDLoss(c, reduction=reduction)
        else:
            self.svdd_loss = SVDDLoss(c, reduction=reduction)

        if nac is True:
            self.mse_loss = torch.nn.MSELoss(reduction=reduction)

    def forward(self, rep, rep2=None, y=None, pred=None):
        """

        :param rep:
        :param rep2: valid when unc is True
        :param y: valid when nac is True
        :param pred: valid when nac is True
        :return:
        """

        if self.unc:
            loss_rep = self.unc_svdd_loss(rep, rep2)
        else:
            loss_rep = self.svdd_loss(rep)

        if self.nac:
            loss_nac = self.mse_loss(pred, y)
            loss = loss_rep + self.alpha * loss_nac
            # print(pred.shape, y.shape, loss_rep, loss_nac, loss)

        else:
            loss = loss_rep

        return loss


class UncSVDDLoss(torch.nn.Module):
    """calibrated uncertainty-aware loss"""
    def __init__(self, c, reduction='mean'):
        super(UncSVDDLoss, self).__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, rep, rep2):
        dis1 = torch.sum((rep - self.c) ** 2, dim=1)
        dis2 = torch.sum((rep2 - self.c) ** 2, dim=1)
        var = (dis1 - dis2) ** 2

        loss = 0.5*torch.exp(torch.mul(-1, var)) * (dis1+dis2) + 0.5*var

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class SVDDLoss(torch.nn.Module):
    def __init__(self, c, reduction='mean'):
        super(SVDDLoss, self).__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, rep):
        loss = torch.sum((rep - self.c) ** 2, dim=1)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


