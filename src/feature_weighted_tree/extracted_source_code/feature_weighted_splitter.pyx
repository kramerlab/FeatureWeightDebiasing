# Own variables
cdef float64_t weight_sum
cdef float64_t current_weight_sum
cdef float64_t feature_weight 
cdef float64_t drawn_probability
cdef float64_t n_possible_features 

# Draw a features at random with feature weights
if draw_with_feature_weights: 
    weight_sum = 0.0
    for i in range(f_i):
        if (n_drawn_constants <= i < n_known_constants) or ((n_known_constants + n_found_constants) <= i):
            feature_weight = feature_weights[features[i]]
            possible_weights[i] = feature_weight
            weight_sum += feature_weight 
        else:
            possible_weights[i] = 0.0
    if weight_sum == 0.0:
            n_possible_features = (n_known_constants - n_drawn_constants) + (f_i - (n_found_constants + n_known_constants))
            weight_sum = 1.0 / n_possible_features
    drawn_probability = rand_uniform(0, 1, random_state)
    current_weight_sum = 0.0
    for i in range(n_drawn_constants, f_i):
        if possible_weights[i] == 0.0:
            continue
        current_weight_sum += (possible_weights[i] / weight_sum)
        if current_weight_sum >= drawn_probability:
            f_j = i
            break
else:
    # Draw a feature at random
    f_j = rand_int(n_drawn_constants, f_i - n_found_constants, random_state)