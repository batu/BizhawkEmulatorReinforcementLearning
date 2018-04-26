import numpy as np

def explore_with_rrt(initial_state, successor, goal_method, projection,
					 video_renderer, max_samples=1000):
	edges = []
	states = [initial_state]
	explore_time = [0]
	projections = [projection(initial_state)]
	available_actions = [set(np.arange(4096))]

	for i in range(max_samples):
		goal = goal_method()

		chosen_index = min(range(len(states)),
						   key=lambda i: np.linalg.norm(projections[i] - goal) * (explore_time[i] + 1) / len(available_actions[i]))

		chosen_state = states[chosen_index]
		selected_action, successor_state = successor(chosen_state, goal,
													 available_actions[chosen_state])
		available_actions[chosen_index].remove(selected_action)
		explore_time[chosen_index] += 1
		available_actions.append(set(np.arange(4096)))
		explore_time.append(0)
		chosen_projection = projections[chosen_index]
		successor_projection = projection(successor_state)
		states.append(successor_state)
		projections.append(successor_projection)
		edges.append((goal, chosen_state, selected_action,
					  successor_state, chosen_projection,
					  successor_projection))

	video_renderer(edges)
	return edges
