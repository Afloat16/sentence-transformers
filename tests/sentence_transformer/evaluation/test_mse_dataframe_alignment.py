from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from sentence_transformers.sentence_transformer.evaluation.mse_from_dataframe import MSEEvaluatorFromDataFrame


@pytest.mark.parametrize(
    "dataframe",
    [
        [],
        [{"en": "", "de": "hallo"}],
        [{"en": "hello", "de": " \n\t"}],
        [{"en": "hello", "de": ""}, {"en": "", "de": "hallo"}],
    ],
)
def test_rejects_language_pair_without_aligned_sentences(dataframe):
    teacher = Mock()
    teacher.encode.return_value = np.empty((0, 2))
    with pytest.raises(ValueError, match="en.*de"):
        MSEEvaluatorFromDataFrame(dataframe, teacher, combinations=[("en", "de")])
    teacher.encode.assert_not_called()


def test_rejects_empty_language_combinations():
    teacher = Mock()
    teacher.encode.return_value = np.empty((0, 2))
    with pytest.raises(ValueError, match="at least one language combination"):
        MSEEvaluatorFromDataFrame([{"en": "hello", "de": "hallo"}], teacher, combinations=[])
    teacher.encode.assert_not_called()


def test_validates_all_combinations_before_encoding():
    teacher = Mock()
    teacher.encode.return_value = np.empty((0, 2))
    with pytest.raises(ValueError, match="en.*fr"):
        MSEEvaluatorFromDataFrame(
            [{"en": "hello", "de": "hallo", "fr": ""}],
            teacher,
            combinations=[("en", "de"), ("en", "fr")],
        )
    teacher.encode.assert_not_called()


def test_partial_translations_keep_language_pair_averaging():
    dataframe = [
        {"en": "en1", "de": "de1", "fr": ""},
        {"en": "en2", "de": "de2", "fr": "fr2"},
        {"en": "en3", "de": "", "fr": ""},
    ]
    teacher_vectors = {"en1": [1.0, 0.0], "en2": [0.0, 1.0]}
    student_vectors = {"de1": [2.0, 0.0], "de2": [0.0, 1.0], "fr2": [0.0, 3.0]}
    teacher = Mock()
    teacher.encode.side_effect = lambda sentences, **kwargs: np.array([teacher_vectors[s] for s in sentences])
    student = Mock()
    student.encode.side_effect = lambda sentences, **kwargs: np.array([student_vectors[s] for s in sentences])

    evaluator = MSEEvaluatorFromDataFrame(
        dataframe, teacher, combinations=[("en", "de"), ("en", "fr")], write_csv=False
    )
    assert evaluator(student) == {"negative_mse": -112.5}
    assert teacher.encode.call_count == 1
    assert [call.args[0] for call in student.encode.call_args_list] == [["de1", "de2"], ["fr2"]]
