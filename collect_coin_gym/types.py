from typing import Literal, NamedTuple, TypeAlias


class Location(NamedTuple):
    r: int
    c: int


Path: TypeAlias = list[Location]


class Move(NamedTuple):
    r: int
    c: int


class ObjectId(NamedTuple):
    border: int
    blue_agent: int
    red_agent: int
    red_coin: int
    green_coin: int
    blue_coin: int


class FieldObject(NamedTuple):
    background: list[Location]
    blue_agent: Location
    red_agent: Location
    red_coin: list[Location]
    green_coin: list[Location]
    blue_coin: list[Location]
    border: list[Location]


Quadrant: TypeAlias = Literal[0, 1, 2, 3, 4]


class Quadrants(NamedTuple):
    first: list[Location]
    second: list[Location]
    third: list[Location]
    fourth: list[Location]


class ObjectSpawnQuadrants(NamedTuple):
    blue_agent: Quadrant
    red_agent: Quadrant
    red_coin: Quadrant
    green_coin: Quadrant
    blue_coin: Quadrant


PeerPolicy: TypeAlias = Literal["random", "stay"]
