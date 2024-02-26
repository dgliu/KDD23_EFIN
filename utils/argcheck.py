import argparse


# Commandline parameter constrains
def check_int_positive(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def check_float_positive(value):
    ivalue = float(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive float value" % value)
    return ivalue


def ratio(s):
    train, valid, test = map(float, s.split(','))
    if train + valid + test != 1:
        raise argparse.ArgumentTypeError("The sum of ratios must equals to one")
    return train, valid, test


def ratio_without_test(s):
    train, valid = map(float, s.split(','))
    if train + valid != 1:
        raise argparse.ArgumentTypeError("The sum of ratios must equals to one")
    return train, valid
