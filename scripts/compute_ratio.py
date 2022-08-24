def compute_ratio(box_ap, mask_ap):
    base_box_ap = 34.57
    base_mask_ap = 29.18
    increase_box_ap_ratio = ((box_ap - base_box_ap) / base_box_ap) * 100
    increase_mask_ap_ratio = ((mask_ap - base_mask_ap) / base_mask_ap) * 100
    print('box_ap的提升比率为：{}%'.format(round(increase_box_ap_ratio, 2)))
    print('mask_ap的提升比率为：{}%'.format(round(increase_mask_ap_ratio, 2)))


def compute_ap(increase_box_ap_ratio, increase_mask_ap_ratio):
    base_box_ap = 34.57
    base_mask_ap = 29.18
    box_ap = ((increase_box_ap_ratio / 100) * base_box_ap) + base_box_ap
    mask_ap = ((increase_mask_ap_ratio / 100) * base_mask_ap) + base_mask_ap

    print('box_ap为：{}'.format(round(box_ap, 5)))
    print('mask_ap为：{}'.format(round(mask_ap, 5)))


if __name__ == '__main__':
    while True:
        # box_ap = eval(input('please input box_ap: '))
        # mask_ap = eval(input('please input mask_ap: '))
        # compute_ratio(box_ap, mask_ap)

        increase_box_ap_ratio = eval(input('please input increase_box_ap_ratio: '))
        increase_mask_ap_ratio = eval(input('please input increase_mask_ap_ratio: '))
        compute_ap(increase_box_ap_ratio, increase_mask_ap_ratio)
