import os
from subprocess import Popen
import time

for i in range(5):
    # p = Popen(['bash', '-c',
    #            "kubectl get datasets.data.fluid.io oss-synthetic-imagenet -o=jsonpath='[{.status.ufsTotal},{.status.cacheStates.cached},{.status.cacheStates.cachedPercentage}, {.status.cacheStates.cacheHitRatio}]' && echo",
    #            "2>&1", "|", "tee", "-a", "/logs/test.log"])
    p = Popen(['bash', '/logs/log_collector.sh', str(i)])
    time.sleep(1)