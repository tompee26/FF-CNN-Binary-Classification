import os
import model

TRAIN = os.path.realpath(os.curdir + "/train")
TEST = os.path.realpath(os.curdir + "/test")
GRAPH = os.path.realpath(os.curdir + "/graph")


def make_hparam_string(learning_rate, extra_fc, use_two_conv):
    conv_param = "conv=2" if use_two_conv else "conv=1"
    fc_param = "fc=2" if extra_fc else "fc=1"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)


def main():
    for learning_rate in [1e-3, 1e-4]:
        for extra_fc in [False, True]:
            hparam = make_hparam_string(learning_rate=learning_rate, extra_fc=extra_fc, use_two_conv=False)
            model.conv_net_model(learning_rate, GRAPH + hparam, TRAIN, TEST)


if __name__ == "__main__":
    main()
