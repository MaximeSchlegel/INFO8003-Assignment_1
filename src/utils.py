def decompose_trajectory(trajectory):
    assert (len(trajectory) - 1) % 3 == 0, "The trajectory is not valid : incorrect length"
    res = []
    for i in range((len(trajectory) - 1) // 3):
        res.append((trajectory[3 * i], trajectory[3 * i + 1], trajectory[3 * i + 2], trajectory[3 * i + 3]))
    return res
