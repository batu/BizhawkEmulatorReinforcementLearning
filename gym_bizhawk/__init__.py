import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='BizHawk-v0',
    entry_point='gym_bizhawk.envs:BizHawkEnv',
)
