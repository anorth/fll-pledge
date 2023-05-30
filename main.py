import dataclasses
import json

from pledge.consts import YEAR, DAY, PEBIBYTE, EXBIBYTE
from pledge.network import MAINNET_APR_2023, NetworkState, BehaviourConfig, SECTOR_LIFETIME_DEFAULT, \
    SECTOR_COMMITMENT_DEFAULT, Onboarding


def main():
    step_size_epochs = 120
    simulation_epochs = 7 * YEAR
    stats_interval_epochs = DAY
    netcfg = dataclasses.replace(
        MAINNET_APR_2023,
        baseline_growth=0.0,
    )
    # Onboarding rate at replacement for current power
    onboarding = Onboarding.constant(netcfg.qa_power / SECTOR_LIFETIME_DEFAULT)
    # onboarding = Onboarding.increasing(netcfg.epoch, 12 * PEBIBYTE / DAY, 1 * PEBIBYTE / DAY / YEAR)
    behaviour = BehaviourConfig(
        sector_commitment_epochs=SECTOR_COMMITMENT_DEFAULT,
        extension_rate=1.0,
        onboarding=onboarding,
        rebase_pledge=True,
    )
    net = NetworkState(netcfg, behaviour, epoch_step=step_size_epochs)

    first_step = net.step_no
    stats = []
    for step in range(first_step, first_step + simulation_epochs // step_size_epochs):
        if step % (stats_interval_epochs // step_size_epochs) == 0:
            stats.append(net.summary())
        net.handle_epochs()

    for s in stats:
        print(json.dumps(s))


if __name__ == '__main__':
    main()
