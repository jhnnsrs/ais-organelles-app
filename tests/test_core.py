"""The numerics on synthetic volumes, no server: the checks `core._checks` yields, as one test."""

from ais_organelles_app import core


def test_the_flight_finds_the_ais_and_its_organelles():
    results = list(core._checks())
    failed = [f"{label}: {detail}" for label, ok, detail in results if not ok]
    assert results and not failed, "\n".join(failed)
