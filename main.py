import dataclasses
import json

from pledge.consts import YEAR, DAY, PEBIBYTE
from pledge.network import MAINNET_APR_2023, NetworkState, BehaviourConfig


def main():
    step_size_epochs = 40
    epochs = 5 * YEAR
    stats_interval_epochs = DAY
    netcfg = dataclasses.replace(
        MAINNET_APR_2023,
        baseline_growth=0.0,
    )
    behaviour = BehaviourConfig(
        extension_rate=1.0,
        # onboarding_daily=5 * PEBIBYTE,
    )
    net = NetworkState(netcfg, behaviour, epoch_step=step_size_epochs)

    first_step = net.step_no
    stats = []
    for step in range(first_step, first_step + epochs // step_size_epochs):
        if step % (stats_interval_epochs // step_size_epochs) == 0:
            stats.append(net.summary())
        net.handle_epochs()

    for s in stats:
        print(json.dumps(s))


if __name__ == '__main__':
    main()
