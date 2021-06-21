from Component import *

dqn_agent = DqnAgent()
state_tracker = StateTracker()
user = User()
emc = Emc()


def run_round(state, warmup=False):
    """
    the implementation of the bot does not have NLU and NLG components. states are already encoded in semantic frames
    and no decoding to output involved.

    Args:
        state:
        warmup: the initial policy model; rather than randomly chosen

    Returns:

    """
    # update action given the state
    agent_action_index, agent_action = dqn_agent.get_action(state, use_rule=warmup)
    # update the ST with the action chosen on this time step
    round_num = state_tracker.update_state_agent(agent_action)
    # simulated user choose an action given the state and action elicited by the agent (dialogue manager)
    user_action, reward, done, success = user.step(agent_action, round_num)
    # check if terminated
    if not done:
        emc.infuse_error(user_action)
    state_tracker.update_state_user(user_action)
    # prepare the next state for the next time step
    next_state = state_tracker.get_state(done)
    # update the ST with new state, agent action, reward
    dqn_agent.add_experience(state, agent_action, reward, next_state, done)

    return next_state, reward, done, success


def episode_reset():
    state_tracker.reset()
    user_action = user.reset()
    emc.infuse_error(user_action)
    state_tracker.update_state_user(user_action)
    dqn_agent.reset()


def warmup_run():
    total_step = 0
    # REVIEW: why need this outer loop? for running more times?
    while total_step != WARMUP_MEM and not dqn_agent.is_memory_full():
        episode_reset()
        done = False
        state = state_tracker.get_state()
        while not done:
            next_state, _, done, _ = run_round(state, warmup=True)
            total_step += 1
            state = next_state


def train_run():
    # REVIEW: read again
    episode = 0
    period_success_total = 0
    success_rate_best = 0.0
    while episode < NUM_EP_TRAIN:
        episode_reset()
        episode += 1
        done = False
        state = state_tracker.get_state()
        while not done:
            next_state, reward, done, success = run_round(state)
            period_reward_total += reward
            state = next_state

        period_success_total += success

        if episode % TRAIN_FREQ == 0:
            success_rate = period_success_total / TRAIN_FREQ
            if success_rate >= success_rate_best and success_rate >= SUCCESS_RATE_THRESHOLD:
                dqn_agent.empty_memory()
                success_rate_best = success_rate

        period_success_total = 0
        dqn_agent.copy()
        dqn_agent.train()


