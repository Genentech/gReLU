import numpy as np

from grelu.data.augment import Augmenter
from grelu.sequence.format import convert_input_type

seq = convert_input_type("ACGTAC", "indices")
label = np.array([0, 1, 2, 3, 4, 5], dtype=np.float32)


def test_augmenter_base():
    aug = Augmenter(
        rc=False, max_seq_shift=0, max_pair_shift=0, n_mutated_seqs=0, seed=0
    )
    assert len(aug) == 1
    assert aug.max_values == [1, 1, 1, 1]
    assert np.allclose(aug.products, [1, 1, 1, 1])

    aug.mode = "serial"
    for i in range(4):
        assert aug._split(i) == [0, 0, 0, 0]
        x = aug(seq=seq, idx=i)
        assert np.allclose(x, seq)
        x, y = aug(seq=seq, label=label, idx=i)
        assert np.allclose(x, seq)
        assert np.allclose(y, label)

    aug.mode = "random"
    for i in range(4):
        assert aug._get_random_idxs() == [0, 0, 0, 0]
        x = aug(seq=seq, idx=i)
        assert np.allclose(x, seq)
        x, y = aug(seq=seq, label=label, idx=i)
        assert np.allclose(x, seq)
        assert np.allclose(y, label)


def test_augmenter_rc():
    aug = Augmenter(
        rc=True,
        max_seq_shift=0,
        max_pair_shift=0,
        n_mutated_seqs=0,
        seed=0,
    )
    assert len(aug) == 2
    assert aug.max_values == [2, 1, 1, 1]
    assert np.allclose(aug.products, [1, 1, 1, 1])

    # Cycle

    aug.mode = "serial"

    expected_seqs = [
        "ACGTAC",
        "GTACGT",
        "ACGTAC",
        "GTACGT",
    ]
    xs = [aug(seq=seq, idx=i) for i in range(4)]
    xs = [convert_input_type(x, "strings") for x in xs]
    assert xs == expected_seqs
    xs, ys = list(zip(*[aug(seq=seq, label=label, idx=i) for i in range(4)]))
    xs = [convert_input_type(x, "strings") for x in xs]
    assert xs == expected_seqs
    assert np.allclose(
        ys,
        [
            [0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0],
            [0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0],
        ],
    )

    # Random
    aug.mode = "random"
    xs = [aug(seq=seq, idx=i) for i in range(4)]
    xs = [convert_input_type(x, "strings") for x in xs]
    assert xs == ["ACGTAC", "GTACGT", "GTACGT", "ACGTAC"]
    xs, ys = list(zip(*[aug(seq=seq, label=label, idx=i) for i in range(4)]))
    xs = [convert_input_type(x, "strings") for x in xs]
    assert xs == ["GTACGT"] * 4
    assert np.allclose(
        ys,
        [
            [5, 4, 3, 2, 1, 0],
            [5, 4, 3, 2, 1, 0],
            [5, 4, 3, 2, 1, 0],
            [5, 4, 3, 2, 1, 0],
        ],
    )


def test_augmenter_seq_shift():
    # Only seq shift
    aug = Augmenter(
        rc=False,
        max_seq_shift=2,
        max_pair_shift=0,
        n_mutated_seqs=0,
        seq_len=2,
    )
    assert len(aug) == 5
    assert aug.max_values == [1, 5, 1, 1]
    assert np.allclose(aug.products, [5, 1, 1, 1])
    assert [aug._split(i) for i in range(5)] == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 2, 0, 0],
        [0, 3, 0, 0],
        [0, 4, 0, 0],
    ]
    expected_seqs = ["AC", "CG", "GT", "TA", "AC"]
    assert [
        convert_input_type(aug(seq=seq, idx=i), "strings") for i in range(5)
    ] == expected_seqs
    pairs = [aug(seq=seq, label=label, idx=i) for i in range(5)]
    assert [convert_input_type(pair[0], "strings") for pair in pairs] == expected_seqs
    assert np.allclose([pair[1] for pair in pairs], [[0, 1, 2, 3, 4, 5]] * 5)


def test_augmenter_pair_shift():
    # Only pair shift
    aug = Augmenter(
        rc=False,
        max_seq_shift=0,
        max_pair_shift=2,
        n_mutated_seqs=0,
        seq_len=2,
        label_len=2,
    )
    assert len(aug) == 5
    assert aug.max_values == [1, 1, 5, 1]
    assert np.allclose(aug.products, [5, 5, 1, 1])
    assert [aug._split(i) for i in range(5)] == [
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 2, 0],
        [0, 0, 3, 0],
        [0, 0, 4, 0],
    ]
    expected_seqs = ["AC", "CG", "GT", "TA", "AC"]
    assert [
        convert_input_type(aug(seq=seq, idx=i), "strings") for i in range(5)
    ] == expected_seqs
    pairs = [aug(seq=seq, label=label, idx=i) for i in range(5)]
    assert [convert_input_type(pair[0], "strings") for pair in pairs] == expected_seqs
    assert np.allclose(
        [pair[1] for pair in pairs], [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
    )


def test_augmenter_mutations():
    # Only mutations
    aug = Augmenter(
        rc=False, max_seq_shift=0, max_pair_shift=0, n_mutated_seqs=2, n_mutated_bases=1
    )
    assert len(aug) == 2
    assert aug.max_values == [1, 1, 1, 2]
    assert np.allclose(aug.products, [2, 2, 2, 1])
    assert [aug._split(i) for i in range(2)] == [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
    ]
    for i in range(2):
        assert np.sum(aug(seq=seq, idx=i) != seq) == 1
        x, y = aug(seq=seq, label=label, idx=i)
        assert np.sum(x != seq) == 1
        assert np.sum(y != label) == 0

    # rc + seq shift
    aug = Augmenter(
        rc=True,
        max_seq_shift=1,
        max_pair_shift=0,
        n_mutated_seqs=0,
        seq_len=3,
    )
    assert len(aug) == 6
    assert aug.max_values == [2, 3, 1, 1]
    assert np.allclose(aug.products, [3, 1, 1, 1])
    assert [aug._split(i) for i in range(6)] == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 2, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 2, 0, 0],
    ]

    expected_seqs = [
        "ACG",
        "CGT",
        "GTA",
        "CGT",
        "ACG",
        "TAC",
    ]
    assert [
        convert_input_type(aug(seq=seq[:-1], idx=i), "strings") for i in range(6)
    ] == expected_seqs
    pairs = [aug(seq=seq, label=label, idx=i) for i in range(6)]
    assert [convert_input_type(pair[0], "strings") for pair in pairs] == expected_seqs
    assert np.allclose(
        [pair[1] for pair in pairs],
        [
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0],
            [5, 4, 3, 2, 1, 0],
            [5, 4, 3, 2, 1, 0],
        ],
    )


def test_augmenter_composite():
    # rc + seq shift + pair_shift
    aug = Augmenter(
        rc=True,
        max_seq_shift=1,
        max_pair_shift=1,
        n_mutated_seqs=0,
        seq_len=1,
        label_len=1,
    )
    assert len(aug) == 18
    assert aug.max_values == [2, 3, 3, 1]
    assert np.allclose(aug.products, [9, 3, 1, 1])
    assert [aug._split(i) for i in range(18)] == [
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 2, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 2, 0],
        [0, 2, 0, 0],
        [0, 2, 1, 0],
        [0, 2, 2, 0],
        [1, 0, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 2, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 2, 0],
        [1, 2, 0, 0],
        [1, 2, 1, 0],
        [1, 2, 2, 0],
    ]
    expected_seqs = [
        "A",
        "C",
        "G",
        "C",
        "G",
        "T",
        "G",
        "T",
        "A",
        "T",
        "G",
        "C",
        "G",
        "C",
        "A",
        "C",
        "A",
        "T",
    ]
    assert [
        convert_input_type(aug(seq=seq, idx=i), "strings") for i in range(18)
    ] == expected_seqs
    pairs = [aug(seq=seq, label=label, idx=i) for i in range(18)]
    assert [convert_input_type(pair[0], "strings") for pair in pairs] == expected_seqs
    assert np.allclose(
        np.concatenate([pair[1] for pair in pairs]).astype(int),
        [
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
        ],
    )
