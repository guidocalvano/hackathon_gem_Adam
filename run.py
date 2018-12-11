import sys
from Runner import Runner
from Learning import Learning


runner = Runner(sys.argv, {
    "resnet": Learning.run_experiment
})
runner.run()

