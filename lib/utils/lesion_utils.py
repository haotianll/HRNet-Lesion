def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, power=0.9, nbb_mult=10, min_lr=None):
    if min_lr is not None:
        coeff = (1 - cur_iters / max_iters) ** power
        lr = (base_lr - min_lr) * coeff + min_lr
    else:
        lr = base_lr * ((1 - float(cur_iters) / max_iters) ** power)

    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr
