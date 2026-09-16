from fly_crossy.env import generate_rows


def test_generated_trains_are_long_and_rail_lanes_are_slower() -> None:
    rail_lanes = []

    for seed_index in range(12):
        rail_lanes.extend(
            lane
            for lane in generate_rows(f"train-world-{seed_index}", 0, 420)
            if lane.kind == "rail"
        )

    assert rail_lanes
    assert all(lane.speed is not None and 10 <= lane.speed <= 12 for lane in rail_lanes)
    assert all(
        len(lane.hazards) == 1
        and lane.hazards[0].kind == "train"
        and lane.hazards[0].size == 18
        for lane in rail_lanes
    )