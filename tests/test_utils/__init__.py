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
