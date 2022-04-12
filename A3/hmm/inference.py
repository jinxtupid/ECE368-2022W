import numpy as np
import graphics
import rover


def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps

    for i in range(0, num_time_steps):
        forward_messages[i] = rover.Distribution()
        backward_messages[i] = rover.Distribution()
        marginals[i] = rover.Distribution()

    # TODO: Compute the forward messages
    # forward initialization
    for i in all_possible_hidden_states:

        obs_model = observation_model(i)
        prob = 1 if (observations[0] is None) else obs_model[observations[0]]

        forward_messages[0][i] = prob * prior_distribution[i]

    # normalize
    forward_messages[0].renormalize()

    # forward recursion
    for i in range(1, num_time_steps):
        observed = observations[i]

        for zi in all_possible_hidden_states:

            sum = 0

            for zi_minus_1 in forward_messages[i - 1]:
                trans_model = transition_model(zi_minus_1)
                sum += forward_messages[i - 1][zi_minus_1] * trans_model[zi]

            obs_model = observation_model(zi)
            prob = 1 if (observed is None) else obs_model[observed]

            if prob * sum != 0:
                forward_messages[i][zi] = prob * sum

        # normalize
        forward_messages[i].renormalize()

    # TODO: Compute the backward messages

    # backward initialization
    size = num_time_steps - 1

    for i in all_possible_hidden_states:
        backward_messages[size][i] = 1

    # backward recursion
    for i in range(1, num_time_steps):
        for zi in all_possible_hidden_states:

            trans_model = transition_model(zi)
            sum = 0

            for zi_plus_1 in backward_messages[size - i + 1]:
                observed = observations[size - i + 1]

                obs_model = observation_model(zi_plus_1)
                prob = 1 if (observed is None) else obs_model[observed]

                sum += prob * trans_model[zi_plus_1] * backward_messages[size - i + 1][zi_plus_1]

            if sum != 0:
                backward_messages[size - i][zi] = sum

        # normalize
        backward_messages[size - i].renormalize()

    # TODO: Compute the marginals
    for i in range(0, num_time_steps):
        sum = 0
        for j in all_possible_hidden_states:
            temp = forward_messages[i][j] * backward_messages[i][j]

            if temp != 0:
                marginals[i][j] = temp
                sum += temp

        for j in marginals[i].keys():
            marginals[i][j] /= sum

    return marginals


def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here

    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps
    w = [None] * num_time_steps
    prev_z = [None] * num_time_steps

    for i in range(0, num_time_steps):
        w[i] = rover.Distribution()

    # initialization
    initial_observed_position = observations[0]
    for i in all_possible_hidden_states:

        obs_model = observation_model(i)
        prob = 1 if (observations[0] is None) else obs_model[observations[0]]

        prior = prior_distribution[i]

        if prob != 0 and prior != 0:
            w[0][i] = np.log(prob) + np.log(prior)

    # recursion
    for i in range(1, num_time_steps):
        prev_z[i] = dict()
        observed = observations[i]

        for zi in all_possible_hidden_states:

            obs_model = observation_model(zi)
            prob = 1 if (observed is None) else obs_model[observed]

            max = -float('inf')

            for zi_minus_1 in w[i - 1]:
                trans_model = transition_model(zi_minus_1)

                if trans_model[zi] != 0:
                    temp = w[i - 1][zi_minus_1] + np.log(trans_model[zi])

                    if prob != 0 and temp > max:
                        prev_z[i][zi] = zi_minus_1
                        max = temp

            if prob != 0:
                w[i][zi] = np.log(prob) + max

    # compute estimated_hidden_states
    size = num_time_steps - 1

    max = -float('inf')
    for zi in w[size]:
        temp = w[size][zi]
        if temp > max:
            estimated_hidden_states[size] = zi
            max = temp

    for i in range(1, num_time_steps):
        estimated_hidden_states[size - i] = prev_z[num_time_steps - i][estimated_hidden_states[num_time_steps - i]]

    return estimated_hidden_states


if __name__ == '__main__':

    enable_graphics = True

    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'

    # load data
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution = rover.initial_distribution()

    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')

    timestep = 30
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])

    # ~z
    correct = 0
    for i in range(0, num_time_steps):
        if hidden_states[i] == estimated_states[i]:
           correct += 1

    print("Viterbi's error: " + str(1 - correct / 100))

    # ^z
    correct = 0
    for i in range(0, num_time_steps):
        max = 0
        estimate = None
        for j in marginals[i]:
            if marginals[i][j] > max:
                max = marginals[i][j]
                estimate = j
        if hidden_states[i] == estimate:
            correct += 1
        else:
            print("Iteration " + str(i) + ":" + str(estimate))

    print("Forward-backward's error: " + str(1- correct/ 100))


    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()

