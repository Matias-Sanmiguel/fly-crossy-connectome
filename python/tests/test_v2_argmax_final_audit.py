from fly_crossy.v2.argmax_final_audit import ratios

def test_ratios():
    c={"meanScore":20.0,"medianScore":10.0,"meanLength":50.0,"reachedStepLimit":2}
    p={"meanScore":100.0,"medianScore":80.0,"meanLength":200.0,"reachedStepLimit":40}
    r=ratios(c,p)
    assert r["meanScoreRatio"]==0.2
    assert r["medianScoreRatio"]==0.125
    assert r["meanLengthRatio"]==0.25
    assert r["stepLimitRatio"]==0.05
