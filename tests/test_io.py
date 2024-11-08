import os

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from grelu.io import read_tomtom
from grelu.io.bed import read_bed
from grelu.io.bigwig import read_bigwig
from grelu.io.fasta import read_fasta
from grelu.io.genome import read_sizes
from grelu.io.motifs import read_meme_file, read_modisco_report
from grelu.sequence.utils import resize

cwd = os.path.realpath(os.path.dirname(__file__))


def test_read_sizes():
    df = read_sizes("hg38")
    assert df.shape == (194, 2)
    assert df.iloc[0].to_dict() == {"chrom": "chr1", "size": 248956422}


def test_read_tomtom():
    # No q-value threshold
    tomtom_dir = os.path.join(cwd, "files", "tomtom")
    df = read_tomtom(tomtom_dir, qthresh=1)
    assert df.shape == (4, 10)

    # q-value threshold

    df = read_tomtom(tomtom_dir, qthresh=0.5)
    assert df.shape == (3, 10)


def test_read_fasta():
    fa_file = os.path.join(cwd, "files", "test.fa.gz")
    assert np.all(read_fasta(fa_file) == ["AAC", "ATG"])


expected_intervals = pd.DataFrame(
    {"chrom": ["chr1", "chr1", "chr2"], "start": [1, 3, 3], "end": [4, 6, 6]}
)
expected_intervals.index = ["0", "1", "2"]


def test_read_bed():
    bed_file = os.path.join(cwd, "files", "test.bed")
    assert_frame_equal(read_bed(bed_file), expected_intervals)


def test_read_bigwig():
    bw_file = os.path.join(cwd, "files", "test.bw")

    # Simple - no aggregation
    output = read_bigwig(expected_intervals.iloc[:2, :], bw_file, aggfunc=None)
    assert output.shape == (2, 1, 3)
    assert np.allclose(output.squeeze(), [[1, 2, 3], [3, 4, 5]])

    # Sum over all bases
    output = read_bigwig(expected_intervals.iloc[:2, :], bw_file, aggfunc="sum")
    assert output.shape == (2, 1, 1)
    assert np.allclose(output.squeeze(), [6, 12])

    # Mean over bin size = 2
    intervals = resize(expected_intervals.iloc[:2, :], 4)
    output = read_bigwig(intervals, bw_file, aggfunc="mean", bin_size=2)
    assert output.shape == (2, 1, 2)
    assert np.allclose(output.squeeze(), [[1.5, 3.5], [3.5, 5.5]])


def test_read_meme_file():
    meme_file = os.path.join(cwd, "files", "test.meme")
    expected_ma0004 = np.array(
        [
            [0.2, 0.95, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.05, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    expected_ma0006 = np.array(
        [
            [0.125, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.333333, 0.0, 0.958333, 0.0, 0.0, 0.0],
            [0.083333, 0.958333, 0.0, 0.958333, 0.0, 1.0],
            [0.458333, 0.041667, 0.041667, 0.041667, 1.0, 0.0],
        ]
    )

    # All motifs
    motifs = read_meme_file(meme_file)
    assert np.all(list(motifs.keys()) == ["MA0004.1 Arnt", "MA0006.1 Ahr::Arnt"])
    assert np.allclose(list(motifs.values())[0], expected_ma0004)
    assert np.allclose(list(motifs.values())[1], expected_ma0006)

    # Specific motifs
    motifs = read_meme_file(meme_file, names=["MA0004.1 Arnt"])
    assert np.all(list(motifs.keys()) == ["MA0004.1 Arnt"])
    assert np.allclose(list(motifs.values())[0], expected_ma0004)

    # Specific number of motifs
    motifs = read_meme_file(meme_file, n_motifs=1)
    assert np.all(list(motifs.keys()) == ["MA0004.1 Arnt"])
    assert np.allclose(list(motifs.values())[0], expected_ma0004)


def test_read_modisco_report():
    modisco_file = os.path.join(cwd, "files", "test_modisco.h5")
    expected_pos_patterns = [
        np.array(
            [
                [
                    0.23381295,
                    0.2206235,
                    0.22302158,
                    0.24460432,
                    0.22302158,
                    0.2470024,
                    0.24940048,
                    0.59232614,
                    0.0971223,
                    0.02278177,
                    0.00359712,
                    0.45203837,
                    0.51558753,
                    0.12230216,
                    0.18585132,
                    0.23741007,
                    0.23261391,
                    0.23141487,
                    0.23741007,
                    0.20023981,
                    0.19784173,
                    0.22182254,
                    0.21582734,
                    0.20623501,
                    0.20263789,
                    0.19304556,
                    0.23860911,
                    0.20263789,
                    0.20383693,
                    0.17865707,
                ],
                [
                    0.2470024,
                    0.27098321,
                    0.29496403,
                    0.22661871,
                    0.27458034,
                    0.29736211,
                    0.40767386,
                    0.11630695,
                    0.04316547,
                    0.01678657,
                    0.00959233,
                    0.06834532,
                    0.15347722,
                    0.13908873,
                    0.17865707,
                    0.20743405,
                    0.2529976,
                    0.28776978,
                    0.25659472,
                    0.25539568,
                    0.25059952,
                    0.24100719,
                    0.26498801,
                    0.25059952,
                    0.25179856,
                    0.27218225,
                    0.25059952,
                    0.27098321,
                    0.24340528,
                    0.25779376,
                ],
                [
                    0.31534772,
                    0.30695444,
                    0.30455635,
                    0.29136691,
                    0.26258993,
                    0.29136691,
                    0.21342926,
                    0.16067146,
                    0.79016787,
                    0.94964029,
                    0.00239808,
                    0.42086331,
                    0.18585132,
                    0.63908873,
                    0.22901679,
                    0.38968825,
                    0.27458034,
                    0.28297362,
                    0.28177458,
                    0.31894484,
                    0.32014388,
                    0.32374101,
                    0.30335731,
                    0.3177458,
                    0.32254197,
                    0.30815348,
                    0.29736211,
                    0.29016787,
                    0.29736211,
                    0.31055156,
                ],
                [
                    0.20383693,
                    0.20143885,
                    0.17745803,
                    0.23741007,
                    0.23980815,
                    0.16426859,
                    0.1294964,
                    0.13069544,
                    0.06954436,
                    0.01079137,
                    0.98441247,
                    0.058753,
                    0.14508393,
                    0.09952038,
                    0.40647482,
                    0.16546763,
                    0.23980815,
                    0.19784173,
                    0.22422062,
                    0.22541966,
                    0.23141487,
                    0.21342926,
                    0.21582734,
                    0.22541966,
                    0.22302158,
                    0.22661871,
                    0.21342926,
                    0.23621103,
                    0.25539568,
                    0.2529976,
                ],
            ]
        ),
        np.array(
            [
                [
                    0.22994652,
                    0.22192513,
                    0.2473262,
                    0.2459893,
                    0.26336898,
                    0.26604278,
                    0.35026738,
                    0.34625668,
                    0.35561497,
                    0.5,
                    0.53342246,
                    0.4157754,
                    0.30347594,
                    0.28475936,
                    0.24331551,
                    0.15106952,
                    0.0828877,
                    0.06818182,
                    0.25668449,
                    0.02005348,
                    0.01203209,
                    0.04411765,
                    0.06417112,
                    0.12566845,
                    0.14973262,
                    0.20989305,
                    0.19518717,
                    0.22593583,
                    0.21791444,
                    0.23930481,
                ],
                [
                    0.25935829,
                    0.28208556,
                    0.27139037,
                    0.25668449,
                    0.2513369,
                    0.24197861,
                    0.22994652,
                    0.23529412,
                    0.25668449,
                    0.20855615,
                    0.15106952,
                    0.15641711,
                    0.16176471,
                    0.22192513,
                    0.2540107,
                    0.32486631,
                    0.45320856,
                    0.61229947,
                    0.08957219,
                    0.13502674,
                    0.00668449,
                    0.06016043,
                    0.01336898,
                    0.20454545,
                    0.3342246,
                    0.27005348,
                    0.28208556,
                    0.26470588,
                    0.27272727,
                    0.26604278,
                ],
                [
                    0.24465241,
                    0.26069519,
                    0.25534759,
                    0.27540107,
                    0.26604278,
                    0.29411765,
                    0.23262032,
                    0.19117647,
                    0.18181818,
                    0.17780749,
                    0.17647059,
                    0.24465241,
                    0.36497326,
                    0.36898396,
                    0.29010695,
                    0.28609626,
                    0.20588235,
                    0.07219251,
                    0.05882353,
                    0.06417112,
                    0.00802139,
                    0.86096257,
                    0.00802139,
                    0.24465241,
                    0.13636364,
                    0.23128342,
                    0.2486631,
                    0.29545455,
                    0.26069519,
                    0.2540107,
                ],
                [
                    0.26604278,
                    0.23529412,
                    0.22593583,
                    0.22192513,
                    0.21925134,
                    0.19786096,
                    0.18716578,
                    0.22727273,
                    0.20588235,
                    0.11363636,
                    0.13903743,
                    0.18315508,
                    0.1697861,
                    0.12433155,
                    0.21256684,
                    0.23796791,
                    0.25802139,
                    0.2473262,
                    0.59491979,
                    0.78074866,
                    0.97326203,
                    0.03475936,
                    0.9144385,
                    0.42513369,
                    0.37967914,
                    0.28877005,
                    0.27406417,
                    0.21390374,
                    0.2486631,
                    0.24064171,
                ],
            ]
        ),
    ]
    expected_neg_patterns = [
        np.array(
            [
                [
                    0.20874904,
                    0.22179586,
                    0.18956255,
                    0.23100537,
                    0.2210284,
                    0.21565618,
                    0.21105142,
                    0.78741366,
                    0.02148887,
                    0.00460476,
                    0.02225633,
                    0.23100537,
                    0.28626247,
                    0.17574827,
                    0.21181888,
                    0.23714505,
                    0.19493477,
                    0.22026094,
                    0.1980046,
                    0.21181888,
                    0.22947045,
                    0.20798158,
                    0.18802763,
                    0.2056792,
                    0.20107444,
                    0.1980046,
                    0.19109747,
                    0.2018419,
                    0.2171911,
                    0.2095165,
                ],
                [
                    0.27705295,
                    0.25786646,
                    0.26630852,
                    0.25863392,
                    0.29777437,
                    0.28012279,
                    0.43668457,
                    0.05372218,
                    0.01688411,
                    0.00997698,
                    0.00844206,
                    0.1105142,
                    0.21565618,
                    0.22179586,
                    0.23944743,
                    0.22640061,
                    0.28319263,
                    0.23714505,
                    0.24251727,
                    0.30391404,
                    0.26630852,
                    0.28242517,
                    0.2647736,
                    0.27858787,
                    0.26554106,
                    0.27705295,
                    0.28012279,
                    0.27935533,
                    0.28089025,
                    0.27551804,
                ],
                [
                    0.30621642,
                    0.32924021,
                    0.33000767,
                    0.31542594,
                    0.29240215,
                    0.34075211,
                    0.23714505,
                    0.09209517,
                    0.94244052,
                    0.97774367,
                    0.01227936,
                    0.55410591,
                    0.30544896,
                    0.43284728,
                    0.30544896,
                    0.34689179,
                    0.29163469,
                    0.31158864,
                    0.33921719,
                    0.29009977,
                    0.3161934,
                    0.34228703,
                    0.33077513,
                    0.31849578,
                    0.32617038,
                    0.31005372,
                    0.31542594,
                    0.30544896,
                    0.30007675,
                    0.3123561,
                ],
                [
                    0.20798158,
                    0.19109747,
                    0.21412126,
                    0.19493477,
                    0.18879509,
                    0.16346892,
                    0.11511896,
                    0.06676899,
                    0.01918649,
                    0.0076746,
                    0.95702226,
                    0.10437452,
                    0.19263239,
                    0.1696086,
                    0.24328473,
                    0.18956255,
                    0.23023791,
                    0.23100537,
                    0.22026094,
                    0.19416731,
                    0.18802763,
                    0.16730622,
                    0.21642364,
                    0.19723715,
                    0.20721412,
                    0.21488872,
                    0.2133538,
                    0.2133538,
                    0.2018419,
                    0.20260936,
                ],
            ]
        ),
        np.array(
            [
                [
                    0.18931584,
                    0.20899719,
                    0.17713215,
                    0.1846298,
                    0.20712277,
                    0.22774133,
                    0.20712277,
                    0.19119025,
                    0.20805998,
                    0.21368322,
                    0.18181818,
                    0.13402062,
                    0.18837863,
                    0.38706654,
                    0.02905342,
                    0.06091846,
                    0.00468604,
                    0.00374883,
                    0.02436739,
                    0.06279288,
                    0.0656045,
                    0.16494845,
                    0.15557638,
                    0.16213683,
                    0.18181818,
                    0.18931584,
                    0.17994377,
                    0.19493908,
                    0.23523899,
                    0.20056232,
                ],
                [
                    0.32427366,
                    0.29428304,
                    0.31865042,
                    0.30365511,
                    0.30365511,
                    0.34582943,
                    0.34770384,
                    0.31021556,
                    0.29990628,
                    0.31958763,
                    0.32239925,
                    0.41330834,
                    0.37019681,
                    0.18556701,
                    0.63074039,
                    0.0393627,
                    0.00843486,
                    0.97282099,
                    0.87253983,
                    0.22961575,
                    0.40393627,
                    0.30459231,
                    0.29896907,
                    0.31865042,
                    0.31865042,
                    0.29615745,
                    0.32239925,
                    0.3158388,
                    0.27085286,
                    0.28491097,
                ],
                [
                    0.24742268,
                    0.24835989,
                    0.26897844,
                    0.26054358,
                    0.26710403,
                    0.21649485,
                    0.23617619,
                    0.2371134,
                    0.26054358,
                    0.24929709,
                    0.26241799,
                    0.21274602,
                    0.20712277,
                    0.25304592,
                    0.13027179,
                    0.01030928,
                    0.00281162,
                    0.01124649,
                    0.0131209,
                    0.12746017,
                    0.33083411,
                    0.15651359,
                    0.20149953,
                    0.19868791,
                    0.2239925,
                    0.2371134,
                    0.25492034,
                    0.23898782,
                    0.23617619,
                    0.26429241,
                ],
                [
                    0.23898782,
                    0.24835989,
                    0.23523899,
                    0.25117151,
                    0.22211809,
                    0.2099344,
                    0.20899719,
                    0.26148079,
                    0.23149016,
                    0.21743205,
                    0.23336457,
                    0.23992502,
                    0.23430178,
                    0.17432052,
                    0.2099344,
                    0.88940956,
                    0.98406748,
                    0.01218369,
                    0.08997188,
                    0.58013121,
                    0.19962512,
                    0.37394564,
                    0.34395501,
                    0.32052484,
                    0.27553889,
                    0.27741331,
                    0.24273664,
                    0.2502343,
                    0.25773196,
                    0.2502343,
                ],
            ]
        ),
    ]
    output = read_modisco_report(modisco_file, trim_threshold=0)
    assert np.all(
        list(output.keys())
        == ["pos_pattern_0", "pos_pattern_1", "neg_pattern_0", "neg_pattern_1"]
    )
    vals = list(output.values())
    assert np.allclose(vals[0], expected_pos_patterns[0])
    assert np.allclose(vals[1], expected_pos_patterns[1])
    assert np.allclose(vals[2], expected_neg_patterns[0])
    assert np.allclose(vals[3], expected_neg_patterns[1])

    output = read_modisco_report(modisco_file, trim_threshold=0, group="neg")
    assert np.all(list(output.keys()) == ["neg_pattern_0", "neg_pattern_1"])
    vals = list(output.values())
    assert np.allclose(vals[0], expected_neg_patterns[0])
    assert np.allclose(vals[1], expected_neg_patterns[1])

    output = read_modisco_report(modisco_file, trim_threshold=0.5)
    assert np.all(
        list(output.keys())
        == ["pos_pattern_0", "pos_pattern_1", "neg_pattern_0", "neg_pattern_1"]
    )
    vals = list(output.values())
    assert np.allclose(vals[0], expected_pos_patterns[0][:, 9:11])
    assert np.allclose(vals[1], expected_pos_patterns[1][:, 19:23])
    assert np.allclose(vals[2], expected_neg_patterns[0][:, 7:11])
    assert np.allclose(vals[3], expected_neg_patterns[1][:, 14:19])

    output = read_modisco_report(modisco_file, trim_threshold=0.5, group="pos")
    assert np.all(list(output.keys()) == ["pos_pattern_0", "pos_pattern_1"])
    vals = list(output.values())
    assert np.allclose(vals[0], expected_pos_patterns[0][:, 9:11])
    assert np.allclose(vals[1], expected_pos_patterns[1][:, 19:23])
