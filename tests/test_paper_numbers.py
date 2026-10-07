"""Step 6.2: the numbers of the manuscript come from maps/paper_numbers.py.

paper/paper_numbers.tex must equal the generator output, every \\Num macro used in the
manuscript must be defined there, and the key numbers keep the values of the state document
(claims, validation, captions), so that a change in the code that moves a number quoted in the
text fails here instead of silently desynchronising the paper.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "maps"))

import paper_numbers as pn  # noqa: E402

MACRO = re.compile(r"\\newcommand\{\\Num([A-Za-z]+)\}\{([^}]*)\}")


@pytest.fixture(scope="module")
def text():
    return pn.render()


@pytest.fixture(scope="module")
def nums(text):
    return {k: v for k, v in MACRO.findall(text)}


def test_file_matches_generator(text):
    assert pn.OUT.read_text() == text, "run: python maps/paper_numbers.py"


def test_macro_names_and_values(nums, text):
    assert len(nums) == text.count("\\newcommand")             # one per line, all parsed
    for k, v in nums.items():
        assert k.isalpha()
        float(v)                                               # every value is a plain number


def test_every_macro_used_in_the_paper_is_defined(nums):
    used = set()
    for f in [ROOT / "paper" / "main.tex", *sorted((ROOT / "paper" / "sections").glob("*.tex")),
              *sorted((ROOT / "maps" / "figures" / "paper").glob("table_*.tex"))]:
        used |= set(re.findall(r"\\Num([A-Za-z]+)", f.read_text()))
    assert used <= set(nums), sorted(used - set(nums))


# values of the state document (claims of section 3, notes 4.10-4.27, captions.md)
PINNED = {
    # claim 1 (notes 3 and 4.25): at most 1.2 % (1.8 % at the lowest coupling), margin 3.7 (2.8)
    "LongShiftMax": "1.2", "LongShiftMaxLow": "1.8", "LongMarginMin": "3.7",
    "LongMarginMinLow": "2.8", "ShiftSt": "1.23", "ShiftPa": "1.19", "ShiftS": "1.07",
    "ShiftPaLow": "1.78", "ShiftStLow": "1.61", "ShiftSMLow": "1.44",
    "LongTtWrMin": "0.01", "LongTtWrMax": "0.10", "LongBetaRatioMin": "0.03",
    "LongBetaRatioMax": "0.27",
    # claim 2 and fig_modal (4.12, 4.15, 4.19)
    "ModalFourTransitMin": "0.838", "ModalFourTransitOver": "19", "HeadTailRatioGammaOne": "2.00",
    "IntermediateRatioMax": "1.38", "NCPeriod": "27.5", "NCPeriodHead": "19.8", "NCIncrease": "39",
    "SongPeriodHeadLight": "40.0", "SongPeriodTailLight": "28.6", "SuFourTransit": "0.952",
    # fig_beta (4.20) and the scope of the rule T_t <~ W_r/4 (step 6.2)
    "BetaFiveBOneMin": "0.11", "BetaFiveBOneMax": "1.66", "BetaFiveBTwoMedian": "0.39",
    "BetaFiveAOneSlope": "0.205", "BetaFiveMinNearDrive": "0.49", "RuleGammaAnyPosition": "1.7",
    # fig_startup (4.16)
    "StartPeakMax": "1.590", "StartPeakAt": "0.87", "StartDOne": "1.574", "StartDTwo": "1.206",
    "StartDFive": "1.077", "StartDampTwo": "1.549", "StartDampFive": "1.496", "StartDampTen": "1.425",
    "StartTravelMax": "1.80", "CollapseEntryEight": "1.1", "CollapseExitEight": "0.3",
    "CollapseEntryFive": "5.2", "CollapseFastMax": "57",
    # validation (claim 3, 4.10, 4.11, 4.13) and verification (4.8)
    "HarrisonSlowMeas": "24.9", "HarrisonSlowModel": "28.4", "HarrisonSlowExcess": "14",
    "HarrisonEmbeddedMax": "14.1", "HarrisonStillMeas": "7.6", "HarrisonCarriageModel": "2.0",
    "HarrisonTransmissionMin": "0.02", "HarrisonTransmissionMax": "0.04",
    "NCArrivalB": "3.74", "NCArrivalC": "5.86",
    "LodewijksPeakDevMin": "-18", "LodewijksPeakDevMax": "0", "LodewijksPeriodMeas": "18.3",
    "LodewijksPeriodModel": "28.5", "LumpedOrder": "2.0",
    # applications (4.27)
    "LoTtReq": "51.5", "LoTt": "21.3", "LoShortest": "101", "LoShortestOverPeriod": "3.6",
    "LoXiLimit": "0.82", "LoGripRatioMin": "4.2", "LoGripRatioMax": "6.1", "LoSineTtFactor": "2.4",
    "SMTtReq": "60", "SMTtReqTail": "22", "SuTtReqTight": "234", "SuTightFactor": "12",
    "SMShortest": "13", "SuShortest": "7.4",
}


@pytest.mark.parametrize("name,value", sorted(PINNED.items()))
def test_pinned_values(nums, name, value):
    assert nums[name] == value


def test_lumped_errors_are_small(nums):
    assert float(nums["LumpedTensionErr"]) < 2e-6 and float(nums["LumpedFreqErr"]) < 5e-5
