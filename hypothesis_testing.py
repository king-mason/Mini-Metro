import unittest
from hypothesis import given, strategies as st
import simulator

from objects import Station, Line, Train, Passenger


class TestSimulator(unittest.TestCase):
    """Hypothesis-based testing for the simulator module."""

    @given(st.integers(min_value=0, max_value=100))
    def test_simulator_with_valid_input(self, value):
        """Test simulator with various integer inputs."""
        result = simulator.run(value)
        self.assertIsNotNone(result)

    @given(st.lists(st.integers()))
    def test_simulator_with_list_input(self, values):
        """Test simulator with list inputs."""
        result = simulator.process(values)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()


    print()
    print("======= UNIT TESTS =======")

    p_square = Passenger("square")
    p_triangle = Passenger("triangle")
    p_circle = Passenger("circle")

    s_square = Station("square", Point(random.randint(1, 10), random.randint(1, 10)))
    s_triangle = Station("triangle", Point(random.randint(1, 10), random.randint(1, 10)))
    s_circle = Station("circle", Point(random.randint(1, 10), random.randint(1, 10)))

    l1 = Line(0, [s_square, s_triangle])
    l2 = Line(1, [s_triangle, s_circle, s_square])

    t1 = Train(l1)
    t2 = Train(l2)

    s_square.spawn_passenger(STATION_TYPES)
    s_square.spawn_passenger(STATION_TYPES)
    
    s_circle.spawn_passenger(STATION_TYPES)
    s_circle.spawn_passenger(STATION_TYPES)
    s_circle.spawn_passenger(STATION_TYPES)
    s_circle.spawn_passenger(STATION_TYPES)

    t2.board_passenger(p_square)
    t2.board_passenger(p_square)
    t2.board_passenger(p_triangle)

    print(t2) # @ triangle
    t2.move()
    print(t2) # @ circle
    t2.move()
    print(t2) # @ square

    print()
    print("Stopped at square station")
    print("Passengers before:", t2.passengers)
    print("Delivered:", t2.stop_at_station())
    print("Passengers after:", t2.passengers)
    print()

    t2.move()
    print(t2) # @ circle

    print()
    print("Stopped at circle station")
    print("Passengers before:", t2.passengers)
    print("Delivered:", t2.stop_at_station())
    print("Passengers after:", t2.passengers)
    print()

    print()

    sim = MiniMetroSim()
    stations = sim.get_new_stations(1000)
    print(stations)