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
    "NCArrivalB": "3.74", "NCArrivalC": "5.86", "NCArrivalBShort": "2.18",
    "LodewijksPeakDevMin": "-18", "LodewijksPeakDevMax": "0", "LodewijksPeriodMeas": "18.3",
    "LodewijksPeriodModel": "28.5", "LumpedOrder": "2.0",
    # applications (4.27)
    "LoTtReq": "51.5", "LoTt": "21.3", "LoShortest": "101", "LoShortestOverPeriod": "3.6",
    "LoXiLimit": "0.82", "LoGripRatioMin": "4.2", "LoGripRatioMax": "6.1", "LoSineTtFactor": "2.4",
    "SMTtReq": "60", "SMTtReqTail": "22", "SuTtReqTight": "234", "SuTightFactor": "12",
    "SMShortest": "13", "SuShortest": "7.4",
    # closed-form solution, Section 3 (step 6.3; notes 4.14 and 4.9 point 1)
    "MassCorrErrTenth": "0.4", "MassCorrErrFundOne": "3.1", "StepOvershootEntry": "2.2",
    # Section 4 and Appendix B (step 6.4; notes 4.31): drive without speed control on
    # Harrison's conveyor, wave-speed checks, Lodewijks and Song cross-comparison, split
    "HarrisonFreeDriveSlow": "14.2", "HarrisonFreeDriveSecond": "7.0",
    "HarrisonDriveMassMeas": "4.8", "HarrisonDriveMassBeltRatio": "2.5",
    "HarrisonDriveMassSecond": "9.2", "HarrisonDashpotNatural": "19",
    "HarrisonDashpotNaturalChange": "0.4", "HarrisonDashpotSoftMin": "1.9",
    "HarrisonDashpotSoftMax": "2.8", "HarrisonSoftSlowMin": "23.9", "HarrisonSoftSlowMax": "26.4",
    "HarrisonSoftZetaMin": "0.08", "HarrisonSoftZetaMax": "0.23", "HarrisonSoftSecondMin": "8.9",
    "HarrisonSoftSecondMax": "9.1", "HarrisonBetaMin": "0.012", "HarrisonBetaMax": "0.024",
    "HarrisonStoredMax": "10", "HarrisonTransit": "7.0", "HarrisonStillSpeed": "0.007",
    "MujaSpeedDiff": "0.9", "SongCcModel": "995.6", "SongCrModel": "1658.7", "SongFRunModel": "224",
    "SongTtMass": "22", "SongTravelTensions": "1.8", "LodewijksLinearGap": "1",
    "LodewijksLinearGapHis": "3", "LodewijksTravelMeanHis": "3.6", "LodewijksTravelMeanModel": "2.6",
    "LodewijksPeriodEmpty": "18.9", "LodewijksPeriodOwnDrive": "16.4", "SplitErr": "2e-5",
    # Sections 5.1-5.2 (step 6.5): four transits of Song (head take-up) and of Nordell and
    # Ciozda, participation formula (4.15), scope of A1 (4.25), shortest Wheatley-Rubel
    # conveyors (4.26); claim 1 confirmed against an independent root of Eq. (chareq_head)
    "SongFourTransit": "45.8", "SongFourTransitRatio": "0.873", "NCFourTransit": "23.4",
    "NCRatioHead": "0.84", "NCRatio": "1.17", "ModalParticipationFormulaErr": "3.1",
    "ModalParticipationMin": "0.41", "ModalParticipationMax": "0.81", "ModalHatched": "7",
    "ModalHatchedMedian": "1.5", "HeadTailRatioGammaTwo": "1.28", "StAOneShift": "3.2",
    "AOneStartRatioMin": "3.3", "WRCLength": "91", "WRCBeta": "2.13", "WRCRatio": "1.38",
    "WRBRatio": "0.84", "BetaFiveMinGammaOneFour": "0.33", "BetaFiveBOneMedian": "0.76",
    "LongBetaMax": "0.205", "WRRatioLowMax": "1.02", "SuRatio": "0.42",
    # Sections 5.3-5.4 and 6 (step 6.6): universal curve (4.16), crawl (4.17), validity and
    # take-up kinematics (4.18, envelope of maps/validity.kinematics), applications (4.27)
    "StartDThree": "1.130", "StartDTen": "1.038", "StartTriangularMax": "1.333",
    "StartParabolicMax": "1.621", "StartTriangularExcess": "26", "StartResOne": "1.273",
    "StartResTwo": "1.054", "StartResFive": "1.008", "CollapseUniformMax": "0.6",
    "CollapseExitFive": "0.4", "PracticeLo": "1.05", "PracticeSi": "11.81",
    "CrawlAbruptMin": "1.17", "CrawlAbruptMax": "1.66", "CrawlRampOneMin": "1.01",
    "CrawlRampOneMax": "1.07", "CrawlTravelRampOneMax": "1.09", "CrawlHoldDampedZero": "1.17",
    "CrawlHoldDampedOne": "1.09", "CrawlHoldDampedTwo": "1.05",
    "ReboundMax": "1.44", "ValidityExitMax": "0.14", "ValidityGripThreeMin": "0.88",
    "ValidityGripThreeMax": "1.28", "ValidityGripSixteenMin": "0.12",
    "ValidityGripSixteenMax": "0.29", "TightMin": "1.80", "TightMax": "2.08", "SlackMin": "0.14",
    "SlackMax": "0.27", "TakeupAccEnvelope": "2.5", "TakeupAccSlow": "0.5",
    "TakeupVelFast": "1.75", "TakeupVelFifth": "1.14", "TakeupVelOne": "0.30",
    "TakeupVelOneFour": "0.08", "SMShortestOverPeriod": "1.0", "SuShortestOverPeriod": "1.2",
    "SMTtReqMax": "83", "SuTtReqBest": "18.1", "AppAccelMarginMin": "38", "SuTtReq": "19.8",
    "SuExitTight": "-114", "LoPeriodChange": "12", "SMPeriodChange": "20", "SMTt": "140",
    "SMReferenceStart": "40.8", "LoGripFactor": "3.0",
}


@pytest.mark.parametrize("name,value", sorted(PINNED.items()))
def test_pinned_values(nums, name, value):
    assert nums[name] == value


def test_lumped_errors_are_small(nums):
    assert float(nums["LumpedTensionErr"]) < 2e-6 and float(nums["LumpedFreqErr"]) < 5e-5
