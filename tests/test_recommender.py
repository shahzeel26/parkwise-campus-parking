from src.recommender import haversine_m

def test_haversine_zero():
    assert haversine_m(-31.98, 115.81, -31.98, 115.81) == 0

def test_haversine_positive():
    assert haversine_m(-31.98, 115.81, -31.99, 115.82) > 0
