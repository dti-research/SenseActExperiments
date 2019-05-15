# Copyright (c) 2019, Danish Technological Institute.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import copy
import numpy as np
import os
import logging
import sys

# Setup logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

def log_function(env, batch_size, shared_returns, log_running, log_dir): 
    """ Process for logging all data generated during runtime.
    Args:
	    env: An instance of ReacherEnv
        batch_size (int): An int representing timesteps_per_batch provided to the TRPO learn function
        shared_returns (dict): A manager dictionary object containing 'episodic returns' and 'episodic lengths'
        log_running (fn): A multiprocessing Value object containing 0/1 - a flag to allow logging while process is running
    """

    old_size = len(shared_returns['episodic_returns']) 
    time.sleep(5.0)
    logging.debug('Started logging process')
    rets = []

    # Create logs directory
    if not os.path.exists(log_dir): os.makedirs(log_dir)

    time_now = time.time()
    csv_file_path = os.path.join(log_dir, str(time_now)+'.csv')
    file = open(csv_file_path, 'a')
    logging.debug('Logging results to: ' + csv_file_path)

    file.write('Episode,Step,Reward,X-Target,Y-Target,Z-Target,X-Current,Y-Current,Z-Current \n') # Header with names for all values logged

    while log_running.value:
        # make a copy of the whole dict to avoid episode_returns and episodic_lengths getting desync
        copied_returns = copy.deepcopy(shared_returns)
        episode = len(copied_returns['episodic_lengths'])

        # Write current values to file
        if not copied_returns['write_lock'] and  len(copied_returns['episodic_returns']) > old_size:
            logging.debug('Writing to log file.')
            returns = np.array(copied_returns['episodic_returns'])
            old_size = len(copied_returns['episodic_returns'])

            file.write(str(episode) + ',' +
                       str(int(episode*100)) + ',' +
                       (str(rets[-1]) if len(rets) else 'NaN') + ',' +
                       str(env._x_target_[2]) + ',' + 
                       str(env._x_target_[1]) + ',' + 
                       str(env._x_target_[0]) + ',' + 
                       str(env._x_[2]) + ',' + 
                       str(env._x_[1]) + ',' + 
                       str(env._x_[0]) + '\n')

            # Calculate rolling average of rewards
            window_size_steps = 5000
            x_tick = 1000

            if copied_returns['episodic_lengths']:
                ep_lens = np.array(copied_returns['episodic_lengths'])
            else:
                ep_lens = batch_size * np.arange(len(returns))
            cum_episode_lengths = np.cumsum(ep_lens)

            if cum_episode_lengths[-1] >= x_tick:
                steps_show = np.arange(x_tick, cum_episode_lengths[-1] + 1, x_tick)

                for i in range(len(steps_show)):
                    rets_in_window = returns[(cum_episode_lengths > max(0, x_tick * (i + 1) - window_size_steps)) *
                                             (cum_episode_lengths < x_tick * (i + 1))]
                    if rets_in_window.any():
                        rets.append(np.mean(rets_in_window))
                        
    file.close()
