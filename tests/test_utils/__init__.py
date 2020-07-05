from warnings import warn

import torch


def print_device_info(device=None):
    # prints device stats. Code from https://stackoverflow.com/a/53374933/12891528
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print('Using device:', device)

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')


def increase_file_limit(new_soft=2 ** 15, new_hard=2 ** 15):
    """ Increase the maximum number of open files. On Linux, this allows spawning more processes/threads. """
    try:
        import resource  # note: local import to avoid ImportError for those who don't have it
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"Increasing file limit - soft {soft}=>{new_soft}, hard {hard}=>{new_hard}")
        return resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, new_soft), max(hard, new_hard)))
    except Exception as e:
        warn(f"Failed to increase file limit: {e}")
