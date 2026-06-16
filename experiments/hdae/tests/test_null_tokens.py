import pytest
from experiments.hdae.hdae.null_tokens import parse_null_levels


def test_parse_null_levels():
    assert parse_null_levels("") == []
    assert parse_null_levels("2,0,2") == [0, 2]
    assert parse_null_levels([1, 0]) == [0, 1]
