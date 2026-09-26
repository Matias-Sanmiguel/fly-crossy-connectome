from fly_crossy.v2.preference_argmax_final import gates, ratios


def test_ratios_match_expected_values() -> None:
    neural = {"meanScore": 20.0, "medianScore": 10.0, "meanLength": 50.0, "reachedStepLimit": 2, "meanInferenceMs": 40.0}
    planner = {"meanScore": 100.0, "medianScore": 100.0, "meanLength": 200.0, "reachedStepLimit": 40}
    relative = ratios(neural, planner)
    assert relative["meanScoreRatio"] == 0.2
    assert relative["medianScoreRatio"] == 0.1
    assert relative["meanLengthRatio"] == 0.25
    assert relative["stepLimitRatio"] == 0.05


def test_gates_are_pure_evaluation_only() -> None:
    relative = {"meanScoreRatio": 0.8, "medianScoreRatio": 0.7, "meanLengthRatio": 0.65, "stepLimitRatio": 0.6}
    neural = {"meanInferenceMs": 45.0}
    assert all(gates(relative, neural).values())
