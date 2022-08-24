def get_latest_epoch_iteration(name='yolact_base_82_459239.pth'):
    lst = name.split('_')
    iteration = lst[-1].split('.')[-2]
    epoch = lst[-2]
    return int(epoch), int(iteration)

    debug = 0
if __name__ == '__main__':
    ab, cd = get_latest_epoch_iteration()
    debug = 0