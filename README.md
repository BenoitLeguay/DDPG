# DDPG
Implementing Deep Deterministic Policy Gradient Algorithm

![result_reward](https://github.com/BenoitLeguay/DDPG/blob/main/.images/algo.png)

http://proceedings.mlr.press/v32/silver14.pdf

### Pendulum-v0

![result_reward](https://github.com/BenoitLeguay/DDPG/blob/main/.images/Screenshot%20from%202020-10-14%2013-23-25.png)



![pendulum_gif](https://github.com/BenoitLeguay/DDPG/blob/main/.images/pendulum.gif)

###### Agent parameters

### 

```python
noise_init = {'mu': 0.0, 'sigma': 0.2, 'action_dim': action_shape}

replay_buffer_init = {'max_len': 100000, 'batch_size': 128}

actor_init = {'action_high': action_high, 
              'action_low': action_low, 
              'network_init': {'i_shape': state_shape, 
                               'l1_shape': 400, 
                               'l2_shape': 300,
                               'o_shape': action_shape
                              },
              'optimizer': {'lr': 1e-3}
             }

critic_init = {'network_init': {'i_shape': state_shape, 
                               'l1_shape': 400, 
                               'l2_shape': 300,
                               'action_shape': action_shape
                              },
              'optimizer': {'lr': 1e-3}
             }


ddpg_init = {
    'seed': seed,
    'action_shape': action_shape,
    'discount_factor': .99,
    'update_target_rate': .995,
    'update_after': 20000,
    'update_every': 50,
    'noise': noise_init,
    'replay_buffer': replay_buffer_init,
    'actor': actor_init,
    'critic': critic_init
}
```

