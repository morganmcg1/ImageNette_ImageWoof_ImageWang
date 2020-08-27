'''
AdaHessian Optimizer

Paper: https://arxiv.org/pdf/2006.00719.pdf

GitHub: https://github.com/amirgholami/adahessian

@article{yao2020adahessian,
  title={ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning},
  author={Yao, Zhewei and Gholami, Amir and Shen, Sheng and Keutzer, Kurt and Mahoney, Michael W},
  journal={arXiv preprint arXiv:2006.00719},
  year={2020}
}
'''

from fastai.basics import *

def average_sqr_diag_hessian(p, sqr_mom, dampening=True, sqr_avg_diag_hessian=None, hutchinson_trace=None, **kwargs):
    if sqr_avg_diag_hessian is None: sqr_avg_diag_hessian = torch.zeros_like(p.grad.data)
    damp = 1-sqr_mom if dampening else 1.
    sqr_avg_diag_hessian.mul_(sqr_mom).addcmul_(hutchinson_trace, hutchinson_trace, value=damp)
    return {'sqr_avg_diag_hessian': sqr_avg_diag_hessian}

def adahessian_step(p, lr, mom, step, sqr_mom, grad_avg, sqr_avg_diag_hessian, hessian_power, eps, **kwargs):
    "Step for AdaHessian with `lr` on `p`"
    debias1 = debias(mom,     1-mom,     step)
    debias2 = debias(sqr_mom, 1-sqr_mom, step)
    if hessian_power < 1:
        p.data.addcdiv_(grad_avg, ((sqr_avg_diag_hessian/debias2).sqrt() ** hessian_power) + eps, value = -lr / debias1)  
    else:
        p.data.addcdiv_(grad_avg, (sqr_avg_diag_hessian/debias2).sqrt() + eps, value = -lr / debias1)    
    return p

@log_args(to_return=True, but_as=Optimizer.__init__)
def AdaHessian(params, lr=0.15, hessian_power=1., hutchinson_trace=None, mom=0.9, sqr_mom=0.999, eps=1e-4, wd=1e-4, decouple_wd=True):
    "A `Optimizer` for AdaHessian"
    cbs = [weight_decay] if decouple_wd else [l2_reg]
    cbs += [partial(average_grad, dampening=True), average_sqr_diag_hessian, step_stat, adahessian_step]
    return Optimizer(params, cbs, lr=lr, mom=mom, sqr_mom=sqr_mom, hessian_power=hessian_power, eps=eps, wd=wd)


@log_args(but='opt')
class AdaHessianWrapper(Optimizer, GetAttr):
    "Wrap `opt` in a AdaHessian optimizer"
    _default='opt'
    def __init__(self, opt, block_length=32, n_acc=1, fp16=False):
        store_attr(self, 'opt,block_length,n_acc')
        self.acc_count=0
        @patch
        def _backward(self:Learner): self.loss.backward(create_graph=True)
        
    def step(self):
        self._accumulate_grads()
        params, gradsH = self._get_params_grad()
        hvs, v = self._get_hessian(params, gradsH)
        hutchinson_trace = self._get_trace(hvs, v)
        for i, (p,pg,state,hyper) in enumerate(self.opt.all_params(with_grad=True)):
            state['hutchinson_trace'] = hutchinson_trace[i]
            for cb in self.opt.cbs: state = self._update(state, cb(p, **{**state, **hyper}))
            self.opt.state[p] = state
            
    def zero_grad(self):
        self.opt.zero_grad()
            
    def clear_state(self):
        self.opt.clear_state()

    def state_dict(self):
        state = self.opt.state_dict()
        
    def clear_state(self):
        self.opt.clear_state()
    
    def load_state_dict(self, sd):
        self.opt.load_state_dict(sd)
    
    def _accumulate_grads(self):
        self.acc_count += 1
        if self.acc_count < self.n_acc: 
            raise CancelBatchException() #skip weight update
        else: self.acc_count=0
    
    def _get_params_grad(self):
        params, gradsH = [], []
        for p,*_ in self.opt.all_params(with_grad=True):
            params.append(p)
            gradsH.append(0. if p.grad is None else p.grad + 0.)
        return params, gradsH
            
    def _get_hessian(self, params, gradsH):
        device = params[0].device
        v = [torch.randint_like(p, high=2, device=device) for p in params]
        for v_i in v: v_i[v_i == 0] = -1
        hvs = torch.autograd.grad(gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=False)
        return hvs, v
    
    def _get_trace(self, hvs, v):
        hutchinson_trace = []
        for hv, vi in zip(hvs, v):
            param_size = hv.size()

            if len(param_size) <= 1:  
                # For 1D tensor, e.g.,, bias, BatchNorm, LayerNorm etc.
                # Usually, you do not need to set spatial aveging for it, i.e., Hessian diagonal block size is 1 here.
                tmp_output = torch.abs(hv * vi)
                hutchinson_trace.append(tmp_output)

                # Of course, you can also use the same way as 2D tensor does to average your 1D tensor. 
                # tmp_output1 = torch.abs((hv * vi + 0.)).view(-1, self.block_length) # faltten to the N times self.block_length
                # tmp_output2 = torch.abs(torch.sum(tmp_output1, dim=[1])).view(-1) / float(self.block_length)
                # tmp_output3 = tmp_output2.repeat_interleave(self.block_length).view(param_size)
                # hutchinson_trace.append(tmp_output3)

            elif len(param_size) == 2: 
                # For 2D tensor, e.g., the matrix in the fully-connected layer.
                # This is a normal case for MLP, Transformer models. 
                # Usually, a spatial averaging needs to be used here to get the best result.
                # If you are not looking for the absolute best config, you may set it to be 1.
                # In all of our experiments, we sill get pretty good performance.
                tmp_output1 = torch.abs((hv * vi + 0.)).view(-1, self.block_length) # faltten to the N times self.block_length
                tmp_output2 = torch.abs(torch.sum(tmp_output1, dim=[1])).view(-1) / float(self.block_length)
                tmp_output3 = tmp_output2.repeat_interleave(self.block_length).view(param_size)
                hutchinson_trace.append(tmp_output3)
            elif len(param_size) == 3:
                # For 3D tensor, e.g., the 1D Conv layer.
                # This layer is usually used for Char-LM.

                # First Way:
                # Usually, you can set it to be the conv kernel size: in more details, for instance, your input/output channels are 20 and your kernel size is 5, 
                # then the 1D Conv kernel is in size 20x20x5, you can average along the final dim, i.e., the block_length = 5
                tmp_output = torch.abs(torch.sum(torch.abs(
                    hv * vi), dim=[2], keepdim=True)) / vi[0, 1].numel() # torch.sum() reduces the dim 2， i.e. the size 5

                # Second way:
                # Of course, you can also use the same self.block_length to average the spatival Hessian of 3D kernel.
                # tmp_output1 = torch.abs((hv * vi + 0.)).view(-1, self.block_length) # faltten to the N times self.block_length
                # tmp_output2 = torch.abs(torch.sum(tmp_output1, dim=[1])).view(-1) / float(self.block_length)
                # tmp_output3 = tmp_output2.repeat_interleave(self.block_length).view(param_size)
                # hutchinson_trace.append(tmp_output3)

            elif len(param_size) == 4:  
                # For 4D tensor, e.g, the 2D Conv layer
                # This layer is usually used for CV tasks.

                # First Way:
                # Usually, you can set it to be the conv kernel size: in more details, for instance, your input/output channels are 256 and your kernel size is 3x3, 
                # then the 2D Conv kernel is in size 20x20x3x3, you can average along the last two dims, , i.e., the block_length = 9
                if vi.size()[1] == 1:
                    vi_div = vi[0, 0]
                else:
                    vi_div = vi[0, 1]
                tmp_output = torch.abs(torch.sum(torch.abs(
                    hv * vi), dim=[2, 3], keepdim=True)) / vi_div.numel() # torch.sum() reduces the dim 2/3.
                hutchinson_trace.append(tmp_output)

                # Second way:
                # Of course, you can also use the same self.block_length to average the spatival Hessian of 4D kernel.
                # tmp_output1 = torch.abs((hv * vi + 0.)).view(-1, self.block_length) # faltten to the N times self.block_length
                # tmp_output2 = torch.abs(torch.sum(tmp_output1, dim=[1])).view(-1) / float(self.block_length)
                # tmp_output3 = tmp_output2.repeat_interleave(self.block_length).view(param_size)
                # hutchinson_trace.append(tmp_output3)
        return hutchinson_trace
    
    def _update(self, state, new=None):
        if new is None: return state
        if isinstance(new, dict): state.update(new)
        return state
    
@delegates(AdaHessian)
def adahessian(p, lr=0.15, n_acc=1, block_length=32, hessian_power=1., mom=0.9, sqr_mom=0.999, eps=1e-4, wd=1e-4, **kwargs):
    "Convenience method for `AdaHessianWrapper` with `Adahessian`"
    return AdaHessianWrapper(AdaHessian(p, lr=lr, **kwargs), n_acc=n_acc, block_length=block_length)