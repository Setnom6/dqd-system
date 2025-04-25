from typing import List


class PredefinedQuantities:
    listGLeft = [
        [[6, 0, 0], [0, 4, 0], [0, 0, 5]],
        [[6, 0, 0], [0, 4, 0], [0, 0, 5]],
        [[6, 0, 0], [0, 4, 0], [0, 0, 5]],
        [[6, 0, 0], [0, -4, 0], [0, 0, 5]],
        [[6, 0, 0], [0, -4, 0], [0, 0, 5]],
        [[6, 0, 0], [0, -4, 0], [0, 0, 5]]
    ]

    listGRight = [
        [[1.02723, 0, 1.8791], [0, 5, 0], [-2.81865, 0, 0.684822]],
        [[-2.92683, 0, -0.439024], [0, 5, 0], [0.658537, 0, -1.95122]],
        [[2.92683, 0, 0.439024], [0, 5, 0], [-0.658537, 0, 1.95122]],
        [[1.02723, 0, 1.8791], [0, 5, 0], [-2.81865, 0, 0.684822]],
        [[-2.92683, 0, -0.439024], [0, 5, 0], [0.658537, 0, -1.95122]],
        [[2.92683, 0, 0.439024], [0, 5, 0], [-0.658537, 0, 1.95122]]
    ]

    @classmethod
    def getSelectedClassCompleteGTensor(cls, geometricClass: str) -> List[List[List[float]]]:
        if geometricClass == "A":
            return [cls.getGFactorClassA("left"), cls.getGFactorClassA("right")]
        elif geometricClass == "B":
            return [cls.getGFactorClassB("left"), cls.getGFactorClassB("right")]
        elif geometricClass == "C":
            return [cls.getGFactorClassC("left"), cls.getGFactorClassC("right")]
        elif geometricClass == "D":
            return [cls.getGFactorClassD("left"), cls.getGFactorClassD("right")]
        elif geometricClass == "E":
            return [cls.getGFactorClassE("left"), cls.getGFactorClassE("right")]
        elif geometricClass == "F":
            return [cls.getGFactorClassF("left"), cls.getGFactorClassF("right")]
        else:
            raise NotImplementedError(f"Geometric class '{geometricClass}' is not implemented.")

    @classmethod
    def getGFactorClassA(cls, side: str) -> List[List[float]]:
        if side == 'left':
            return cls.listGLeft[0]
        elif side == 'right':
            return cls.listGRight[0]
        else:
            raise ValueError(f"Invalid side '{side}'. Expected 'left' or 'right'.")

    @classmethod
    def getGFactorClassB(cls, side: str) -> List[List[float]]:
        if side == 'left':
            return cls.listGLeft[1]
        elif side == 'right':
            return cls.listGRight[1]
        else:
            raise ValueError(f"Invalid side '{side}'. Expected 'left' or 'right'.")

    @classmethod
    def getGFactorClassC(cls, side: str) -> List[List[float]]:
        if side == 'left':
            return cls.listGLeft[2]
        elif side == 'right':
            return cls.listGRight[2]
        else:
            raise ValueError(f"Invalid side '{side}'. Expected 'left' or 'right'.")

    @classmethod
    def getGFactorClassD(cls, side: str) -> List[List[float]]:
        if side == 'left':
            return cls.listGLeft[3]
        elif side == 'right':
            return cls.listGRight[3]
        else:
            raise ValueError(f"Invalid side '{side}'. Expected 'left' or 'right'.")

    @classmethod
    def getGFactorClassE(cls, side: str) -> List[List[float]]:
        if side == 'left':
            return cls.listGLeft[4]
        elif side == 'right':
            return cls.listGRight[4]
        else:
            raise ValueError(f"Invalid side '{side}'. Expected 'left' or 'right'.")

    @classmethod
    def getGFactorClassF(cls, side: str) -> List[List[float]]:
        if side == 'left':
            return cls.listGLeft[5]
        elif side == 'right':
            return cls.listGRight[5]
        else:
            raise ValueError(f"Invalid side '{side}'. Expected 'left' or 'right'.")
