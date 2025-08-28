import torch
def grad_distance(g1, g2, FLAGS):
    if FLAGS.top_grad_percentage < 1.0 and g1.dim() >=2 : # only do this for 2D or higher
        top_k = int(FLAGS.top_grad_percentage * g2.numel())
        g1 = g1.flatten()
        g2 = g2.flatten()
        # import ipdb;ipdb.set_trace()
        indices = torch.topk(g2.abs(), top_k)[1] 
        g1 = g1[indices]
        g2 = g2[indices]

    if FLAGS.distance_function.lower() == 'cosine':
        # use 1-costine similarity
        return 1 - torch.nn.functional.cosine_similarity(g1.reshape(1,-1), g2.reshape(1,-1))
    elif FLAGS.distance_function.lower() == 'l2':
        return torch.nn.functional.mse_loss(g1, g2)
    elif FLAGS.distance_function.lower() == 'l1':
        return torch.nn.functional.l1_loss(g1, g2)
    elif FLAGS.distance_function.lower() == 'cosine+l2':
        return FLAGS.distance_function_weight * (1 - torch.nn.functional.cosine_similarity(g1.reshape(1,-1), g2.reshape(1,-1)) ) +\
                (1- FLAGS.distance_function_weight) * torch.nn.functional.mse_loss(g1, g2)
    else:
        raise NotImplementedError

# ------------------------------------------------------------------------------
# Meta loss
# ------------------------------------------------------------------------------
def meta_loss(output, targets, output_sizes, target_sizes, dldw_targets,  params_to_match, loss_func, FLAGS):
    loss = loss_func(output, targets)
    dldws = torch.autograd.grad(loss, params_to_match, create_graph=True)
    # loss = ((dldw-dldw_target)**2).mean() #MSE
    #loss = 1 - torch.nn.functional.cosine_similarity(dldw.reshape(1,-1), dldw_target.reshape(1,-1))    
    loss = 0 
    for dldw, dldw_target in zip(dldws, dldw_targets):
        #loss += torch.nn.functional.mse_loss(dldw, dldw_target)
        loss += grad_distance(dldw, dldw_target, FLAGS)

    return loss,dldws

# ------------------------------------------------------------------------------
def mae_tensor(x, y):
    # measure the mean absolute error between two tensors
    return torch.mean(torch.abs(x-y))
def tv_norm( x):
    # Compute differences along the y-axis
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    # Compute differences along the x-axis
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    # Compute total variation
    tv = torch.sum(dx) + torch.sum(dy)
    # Scale by the strength parameter
    return tv

