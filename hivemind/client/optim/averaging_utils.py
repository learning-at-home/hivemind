from itertools import chain
from typing import Sequence, Dict
import torch
from hivemind.client.averaging import DecentralizedAverager
from hivemind.utils import nested_flatten, nested_pack


# TODO: Move to averaging
class ParameterAndGradientAverager(DecentralizedAverager):
    """
    A DecentralizedAverager that averages trainable parameters & gradients and can load optimizer state from peers
    :param opt: base pytorch optimizer to be shared in load_state_from_peers
    """
    def __init__(self, opt: torch.optim.Optimizer, global_step: int = 0, **kwargs):
        initialize_optimizer_state(opt)
        self.opt, self.global_step = opt, global_step
        self.optimized_parameters = tuple(param for param_group in opt.param_groups
                                          for param in param_group['params'])

        # averaged tensors are: (*model parameters, *model gradients)
        averaged_tensors = tuple(param.detach().cpu().float().clone() for param in self.optimized_parameters)
        averaged_tensors += tuple(torch.zeros_like(tensor) for tensor in averaged_tensors)
        super().__init__(averaged_tensors=averaged_tensors, **kwargs)

    def get_current_state(self):
        """
        Get current model/optimizer state and when requested by a newbie peer. executed in the host process.
        :returns: a tuple of (serializable_small_metadata, sequence of torch tensors)
        """
        with torch.no_grad():
            optimized_parameters = tuple(param for param_group in self.opt.param_groups
                                         for param in param_group['params'])
            optimizer_metadata, optimizer_tensors = dump_optimizer_state(self.opt)

        metadata = dict(group_bits=self.get_group_bits(), optimizer_metadata=optimizer_metadata)
        return metadata, list(chain(optimized_parameters, optimizer_tensors))

    def load_state_from_peers(self, **kwargs):
        """ Attempt to download the latest optimizer state from peers and update trainer parameters/statistics """
        loadad_state = super().load_state_from_peers(**kwargs)
        if loadad_state is None:
            return

        metadata, flat_tensors = loadad_state
        num_params = len(list(self.optimized_parameters))
        optimized_parameters, opt_tensors = flat_tensors[:num_params], flat_tensors[num_params:]
        with torch.no_grad():
            for local_param, loaded_param in zip(self.optimized_parameters, optimized_parameters):
                local_param[...] = loaded_param
            load_optimizer_state(self.opt, metadata['optimizer_metadata'], opt_tensors)

        collaboration_step = metadata['step']
        while self.global_step < collaboration_step:
            self.global_step += 1
            #TODO update scheduler


    def step(self, weight):
        with torch.no_grad(), self.averager.get_tensors() as averaged_tensors:
            assert len(averaged_tensors) == len(local_tensors)
            for averaged_tensor, local_tensor in zip(averaged_tensors, local_tensors):
                averaged_tensor[...] = local_tensor.detach().cpu().float() * weight

        self.averager.step(timeout=self.averaging_step_timeout)

        # we averaged parameters multiplied by grad scale (aka weights). Time to compensate for that
        # by dividing weights by the sum of grad scales over the entire group.
        sum_of_weights = sum(info['weight'] for info in group_infos.values()
                             if isinstance(info.get('weight'), float))

        normalization_coefficient = (len(group_infos) / sum_of_weights) if sum_of_weights > 0 else 1.0
        with torch.no_grad(), self.averager.get_tensors() as averaged_tensors:
            assert len(averaged_tensors) == len(local_tensors)
            for averaged_tensor, local_tensor in zip(averaged_tensors, local_tensors):
                averaged_tensor *= normalization_coefficient
                local_tensor[...] = averaged_tensor.to(dtype=local_tensor.dtype, device=local_tensor.device)

        logger.info(f"Averaging with peers: done! [group size = {len(group_infos)}, loss = {average_loss:.3f}]")


def initialize_optimizer_state(opt: torch.optim.Optimizer):
    for param_group in opt.param_groups:
        for param in param_group['params']:
            if param.grad is None:
                (0 * param.sum()).backward()
    opt.step()


def dump_optimizer_state(opt: torch.optim.Optimizer):
    with torch.no_grad():
        flat_metadata, flat_tensors = [], []
        for elem in nested_flatten(opt.state_dict()):
            if isinstance(elem, torch.Tensor):
                flat_metadata.append(dict(type='tensor', index=len(flat_tensors)))
                flat_tensors.append(elem.cpu())
            else:
                flat_metadata.append(dict(type='value', value=elem))
        return flat_metadata, flat_tensors


def load_optimizer_state(optimizer: torch.optim.Optimizer, flat_metadata: Dict, flat_tensors: Sequence[torch.Tensor]):
    flat_optimizer_state = []
    for elem in flat_metadata:
        if elem.get('type') == 'tensor' and isinstance(elem.get('index'), int):
            flat_optimizer_state.append(flat_tensors[elem['index']])
        elif elem.get('type') == 'value' and 'value' in elem:
            flat_optimizer_state.append(elem['value'])
    with torch.no_grad():
        return optimizer.load_state_dict(nested_pack(flat_optimizer_state, structure=optimizer.state_dict()))
