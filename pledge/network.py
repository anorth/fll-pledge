import math
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple, Callable

from .consts import DAY, SECTOR_SIZE, YEAR, PEBIBYTE

SUPPLY_LOCK_TARGET = 0.30
INITIAL_PLEDGE_PROJECTION_PERIOD = 20 * DAY

# Reward at epoch = initial reward * (1-r)^epochs
REWARD_DECAY = 1 - math.exp(math.log(1 / 2) / (6 * YEAR))
# Baseline at epoch = initial baseline * (1+b)^epochs
BASELINE_GROWTH = math.exp(math.log(2) / YEAR) - 1

# Interval between vesting chunks
VESTING_INTERVAL = 2880
# Number of intervals over which vesting occurs
VESTING_PERIOD_INTERVALS = 180

# Uniform duration for which sectors are committed/extended
SECTOR_COMMITMENT_DEFAULT = 540 * DAY
# Maximum lifetime for sectors
SECTOR_LIFETIME_DEFAULT = 5 * YEAR
# Fraction of expiring sectors that are extended, at each opportunity
SECTOR_EXTENSION_RATE_DEFAULT = 0.7


@dataclass
class NetworkConfig:
    epoch: int
    qa_power: int
    raw_byte_power: int
    baseline_power: int
    epoch_reward: float
    baseline_growth: float
    reward_decay: float
    circulating_supply: float
    pledge_locked: float
    reward_locked: float


# 2023-04-01
MAINNET_APR_2023 = NetworkConfig(
    epoch=2733360,
    qa_power=22436033270683107000,
    raw_byte_power=14846032093347054000,
    baseline_power=17550994139680311000,
    epoch_reward=5 * 16.7867382504675,
    baseline_growth=BASELINE_GROWTH,
    reward_decay=REWARD_DECAY,
    circulating_supply=456583469.869076,
    pledge_locked=122732221.313155,
    reward_locked=17403179.4812188,
)


class Onboarding:
    """Onboarding function builders"""
    @staticmethod
    def constant(per_epoch: int) -> Callable[[int, int], int]:
        """Returns an onboarding function of a fixed daily amount."""
        def onboarding(_epoch: int, _power: int) -> int:
            return per_epoch
        return onboarding

    @staticmethod
    def proportion(proportion: float) -> Callable[[int, int], int]:
        """Returns an onboarding function of a fixed proportion of current power."""
        def onboarding(_epoch: int, power: int) -> int:
            return int(proportion * power)
        return onboarding

    @staticmethod
    def linear(start_epoch: int, base: int, slope: int) -> Callable[[int, int], int]:
        """Returns an onboarding function that changes linearly with time."""
        def onboarding(epoch: int, _power: int) -> int:
            return base + (epoch - start_epoch) * slope
        return onboarding

@dataclass
class BehaviourConfig:
    # Duration for which to commit/extend sectors
    sector_commitment_epochs: int = SECTOR_COMMITMENT_DEFAULT
    # Maximum total sector lifetime
    sector_lifetime_epochs: int = SECTOR_LIFETIME_DEFAULT
    # Fraction of expiring commitments to extend at each opportunity
    extension_rate: float = SECTOR_EXTENSION_RATE_DEFAULT
    # New onboarding function of (epoch, current power) -> new power per epoch.
    onboarding: Callable[[int, int], int] = Onboarding.constant(0)
    # Whether to reduce pledge to network requirement, if lower, when extending.
    rebase_pledge: bool = False


class NetworkState:
    """
    Models basic network behaviour under simple onboarding and sector extension assumptions.
    The intention is a model to guide intuition and explore alternatives, rather than
    a prediction of mainnet future. There are some simplifications, especially in the initial state.
    """
    def __init__(self, cfg: NetworkConfig, behaviour: BehaviourConfig, epoch_step: int):
        assert epoch_step < VESTING_INTERVAL
        assert VESTING_INTERVAL % epoch_step == 0
        self.behaviour = behaviour

        self.step_size = epoch_step
        self.step_no: int = cfg.epoch // epoch_step
        self.power_baseline: int = cfg.baseline_power
        self.epoch_reward: float = cfg.epoch_reward
        self.baseline_growth: float = cfg.baseline_growth
        self.reward_decay: float = cfg.reward_decay
        self.circulating_supply: float = cfg.circulating_supply

        self.reward_locked: float = 0  # initialised below
        # Reward vesting amounts indexed by step.
        self._reward_vesting: dict[int, float] = defaultdict(float)
        # This flat vesting isn't exactly the right shape to match network state,
        # given block reward decay.
        self.lock_reward(self.step_no, cfg.reward_locked)

        self.power: int = 0
        self.pledge_locked: float = 0
        # Scheduled expiration of power, by step.
        self._expirations: dict[int, list[SectorBunch]] = defaultdict(list[SectorBunch])
        # Assign expiration & end-of-life epochs to initial power, assuming uniform onboarding.
        # This isn't a super accurate model of the real distribution of expirations and lifetimes.
        bunch_power = cfg.qa_power // self.step_no
        bunch_pledge = cfg.pledge_locked / self.step_no
        for onboard_step in range(self.step_no):
            termination = onboard_step + self.behaviour.sector_lifetime_epochs // self.step_size
            expiration = onboard_step + self.behaviour.sector_commitment_epochs // self.step_size
            while expiration < self.step_no:
                expiration += self.behaviour.sector_commitment_epochs // self.step_size
            self.power += bunch_power
            self.pledge_locked += bunch_pledge
            self._expirations[expiration].append(SectorBunch(bunch_power, bunch_pledge, termination))

    def summary(self, rounding=0):
        total_locked = self.pledge_locked + self.reward_locked
        locked_target = total_locked / self.circulating_supply
        epoch = self.step_no * self.step_size
        return {
            'day': epoch // DAY,
            # 'step': self.step_no,
            'epoch': epoch,
            'power': self.power,
            'epoch_reward': self.epoch_reward,
            'baseline': self.power_baseline,
            'circulating_supply': round(self.circulating_supply, rounding),
            'pledge_locked': round(self.pledge_locked, rounding),
            'reward_locked': round(self.reward_locked, rounding),
            'locked_target': round(locked_target, ndigits=4),
            'sector_pledge': self.initial_pledge_for_power(SECTOR_SIZE),
        }

    def handle_epochs(self):
        epoch = self.step_no * self.step_size
        # Emit and vest rewards
        self.lock_reward(self.step_no, self.epoch_reward * self.step_size)

        # Vest rewards
        vesting_now = self._reward_vesting.pop(self.step_no, 0.0)
        if vesting_now > 0.0:
            self.reward_locked -= vesting_now
            self.circulating_supply += vesting_now

        # Expire old power & pledge
        for bunch in self._expirations.pop(self.step_no, []):
            self.power -= bunch.power
            self.pledge_locked -= bunch.pledge
            # Extend some of the expiring power, if possible
            extend_power = int(bunch.power * self.behaviour.extension_rate)
            extend_pledge = self.initial_pledge_for_power(extend_power)
            if not self.behaviour.rebase_pledge:
                extend_pledge = max(bunch.pledge * self.behaviour.extension_rate, extend_pledge)
            if bunch.termination_step > self.step_no:
                self.pledge_sectors(self.step_no, extend_power, extend_pledge, bunch.termination_step)

        # Onboard new power
        new_power = self.behaviour.onboarding(epoch, self.power) * self.step_size
        self.pledge_sectors(self.step_no, new_power, self.initial_pledge_for_power(new_power),
            self.step_no + self.behaviour.sector_lifetime_epochs // self.step_size)

        # Update reward and baseline functions
        for _ in range(self.step_size):
            self.epoch_reward *= (1 - self.reward_decay)
            self.power_baseline *= (1 + self.baseline_growth)
        self.step_no += 1

    def initial_pledge_for_power(self, power: int) -> float:
        """The initial pledge requirement for an incremental power addition."""
        storage = self.expected_reward_for_power(power, INITIAL_PLEDGE_PROJECTION_PERIOD)
        consensus = self.circulating_supply * power * SUPPLY_LOCK_TARGET / max(self.power,
            self.power_baseline)
        return storage + consensus

    def expected_reward_for_power(self, power: int, duration: int, decay=REWARD_DECAY) -> float:
        """Projected rewards for some power over a period, taking reward decay into account."""
        # Note this doesn't use alpha/beta filter estimate or take baseline rewards into account.
        if self.power <= 0:
            return self.projected_reward(self.epoch_reward, duration, decay)
        return self.projected_reward(self.epoch_reward * power / self.power, duration, decay)

    def projected_reward(self, epoch_reward: float, duration: int, decay=REWARD_DECAY) -> float:
        """Projects a per-epoch reward into the future, taking decay into account"""
        return epoch_reward * sum_over_exponential_decay(duration, decay)

    def lock_reward(self, step: int, amount: float):
        self.reward_locked += amount

        each_vest = amount / VESTING_PERIOD_INTERVALS
        interval = VESTING_INTERVAL // self.step_size
        e = (step // interval) * interval + interval
        while amount > 0.0:
            vest_amount = min(each_vest, amount)
            self._reward_vesting[e] += vest_amount
            amount -= vest_amount
            e += interval

    def pledge_sectors(self, step: int, power: int, pledge: float, lifetime_end: int):
        assert lifetime_end >= step
        self.power += power
        self.pledge_locked += pledge
        expiration = min(step + self.behaviour.sector_commitment_epochs // self.step_size,
            lifetime_end)
        self._expirations[expiration].append(SectorBunch(power, pledge, lifetime_end))


class SectorBunch(NamedTuple):
    power: int
    pledge: float
    termination_step: int


def sum_over_exponential_decay(duration: int, decay: float) -> float:
    # SUM[(1-r)^x] for x in 0..duration
    return (1 - math.pow(1 - decay, duration) + decay * math.pow(1 - decay, duration)) / decay
