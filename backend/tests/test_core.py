import math, numpy as np
from app import vision, detect, llm
def test_rppg():
    buf = [(i / 30, math.sin(2 * math.pi * 1.2 * i / 30) + .1 * np.random.randn()) for i in range(450)]
    bpm, c = vision.rppg_bpm(buf); assert abs(bpm - 72) < 3 and c > .5
def test_ppe():
    s = detect.ppe_status([{"label": "no_helmet", "conf": .8}, {"label": "glove", "conf": .7}], ["helmet", "gloves", "shoes"])
    assert s == {"helmet": "missing", "gloves": "ok", "shoes": "not_visible"}
def test_phone():
    o = [{"label": "cell phone", "conf": .9, "box": [.4, .4, .5, .5]}]
    assert detect.phone_in_hand(o, [{"pts": [[.45, .5]]}], None) == (True, True)
    assert detect.phone_in_hand(o, [{"pts": [[.9, .9]]}], None) == (False, True)
def test_gesture_fingers():
    open_hand = [(.5, .9)] + [(.5, .9 - .02 * i) for i in range(1, 21)]      # tips farther from wrist than joints
    fist = [(.5, .9)] + [(.5, .9 - .02 * (i if i % 4 == 1 else 1)) for i in range(1, 21)]
    assert vision.finger_states(open_hand)["index"] and not vision.finger_states(fist)["index"]
def test_compose_fallback():
    assert llm.compose(["I_NEED", "WATER"]).lower().startswith("i need water")
