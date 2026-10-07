import torch
from torch.nn import functional as F


def loss_function(output, batch, ranking_weight=.2, field_weight=1e-5, auxiliary_weight=.1):
    # FP32 likelihood math even when the encoder runs under mixed precision.
    mu, sigma, y = output["mu"].float(), output["sigma"].float(), batch["y"].float()
    weight = batch.get("sample_weight", torch.ones_like(y)).float()
    nll = ((.5*((mu-y)/sigma).square()+sigma.log())*weight).mean()
    huber = (F.huber_loss(mu, y, delta=1., reduction='none')*weight).mean()
    # All within-assay pairs; exclude self-pairs, ties, and repeated samples.
    delta = y[:, None]-y[None, :]
    mask = (batch["group"][:, None] == batch["group"][None, :]) & (delta.abs() > 1e-6)
    mask &= batch["row_id"][:, None] != batch["row_id"][None, :]
    mask &= torch.triu(torch.ones_like(mask), diagonal=1).bool()
    ranks = F.softplus(-(mu[:, None]-mu[None, :])*delta.sign())
    pair_weight = torch.sqrt(weight[:, None]*weight[None, :])
    ranking = (ranks*pair_weight)[mask].mean() if mask.any() else mu.sum()*0
    field = output["field"].float().square().mean(-1)
    regularizer = (field*batch["node_mask"]).sum()/batch["node_mask"].sum().clamp_min(1)
    aux_mask = batch["aux_mask"] & batch["node_mask"][..., None]
    aux = F.huber_loss(output["physical_proxies"].float()[aux_mask], batch["aux_target"].float()[aux_mask]) if aux_mask.any() else mu.sum()*0
    total = nll+.1*huber+ranking_weight*ranking+field_weight*regularizer+auxiliary_weight*aux
    return total, {"nll": nll.detach(), "huber": huber.detach(), "ranking": ranking.detach(),
                   "field": regularizer.detach(), "auxiliary": aux.detach()}
