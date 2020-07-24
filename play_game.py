from DQN import *

name_of_weights_to_save_and_load = 'with_r_with_t_soft.dat'
total_rewards = []
for i in range(1):
    total_reward = cartpole_play(name_of_weights_to_save_and_load)
    total_rewards.append(total_reward)
print('#' * 80)
print(f'Average: {np.mean(total_rewards)}')

# final_net = Network(network_params, device).to(device)
# final_net.load_state_dict(torch.load('best.dat'))
# final_net.eval()
#
# obs = env.reset()
# done = False
# for i in range(30000):
#
#     q_values = final_net(torch.tensor([obs]))
#     _, action = torch.max(q_values, dim=1)
#     action = int(action.item())
#
#     obs, rew, done, info = env.step(action)
#     env.render()
#     # print(rew)
#     if done:
#         obs = env.reset()
# env.close()