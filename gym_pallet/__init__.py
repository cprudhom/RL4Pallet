from gym.envs.registration import register

register(
    id='pallet-v0',
    entry_point='gym_pallet.envs:PalletEnv',
)
register(
    id='pallet-v1',
    entry_point='gym_pallet.envs:PalletEnv2',
)
# register(
#     id='pallet-hard-v0',
#     entry_point='gym_pallet.envs:PalletHardEnv',
# )