import sys
from Runner import Runner
from Learning import Learning


def run(argv):
    runner = Runner(argv, {
        "resnet": Learning.run_experiment
    })
    runner.run()


if __file__ == '__main__':
    run(sys.argv)
