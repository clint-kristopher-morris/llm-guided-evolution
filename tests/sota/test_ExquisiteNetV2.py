import subprocess

def test_train():
    output = subprocess.check_output(['uv', 'run', 'sota/ExquisiteNetV2/train.py', '-bs', '216', '-network', "network", '-data', './cifar10', '-end_lr', '0.001', '-seed', '21', '-val_r', '0.2'])
    print(output)


if __name__ == '__main__':
    test_train()