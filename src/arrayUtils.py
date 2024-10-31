def zeros(length: int) -> list:
    """
    returns an array of zeros with the provided length
    """
    array = []
    i = 0

    while i < length:
        array.append(0)
        i += 1

    return array


def ones(length: int) -> list:
    """
    returns an array of ones with the provided length
    """
    array = []
    i = 0

    while i < length:
        array.append(1)
        i += 1

    return array


def contains(array: list, element: float) -> bool:
    for x in array:
        if element == x:
            return True
    return False
