"""Tests for the reference list and data table filter.

The main point is that authentic prose remains intact regardless of how citation-rich it appears, 
since trend papers are dense in years. Only text with low prose density is tested for compliance 
with citation and table rules.
"""
from clean import reference_like, stopword_fraction
from config import CleanConfig

CFG = CleanConfig()


def test_prose_kept_even_when_citation_dense():
    prose = ("The model shows that warming increases the frequency of extreme "
             "precipitation events across the region and this trend continues "
             "through the century according to the simulations we ran.")
    assert reference_like(prose, CFG) is False


def test_bibliography_dropped():
    biblio = ("Smith J. 2019. doi:10.1/x. Jones A. et al 2020. doi:10.2/y. "
              "Lee K. et al 2018 (2018). Brown 2021 (2021). doi:10.3/z et al et al")
    assert reference_like(biblio, CFG) is True


def test_stopword_fraction_bounds():
    assert stopword_fraction("") == 0.0
    assert 0.0 <= stopword_fraction("the and of in a model warming ocean") <= 1.0
