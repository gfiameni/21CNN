import numpy as np

class Database:
    def __init__(self, BoxesPath, ParametersPath, Redshifts, BoxType = "delta_T", BoxRes = 200, BoxSize = 300, WalkerID = 0, WalkerSteps = 10000):
        self.BoxesPath = BoxesPath
        self.ParametersPath = ParametersPath
        self.BoxType = BoxType
        self.BoxRes = BoxRes
        self.BoxSize = BoxSize
        self.WalkerID = WalkerID
        self.Redshifts = Redshifts
        self.WalkerSteps = WalkerSteps

        self.NumBoxes = len(Redshifts) - 1

    def CreateFilepath(self, RedshiftIndex, WalkerIndex):
        filepath = f"{self.BoxesPath}/{self.BoxType}_{self.WalkerID:.6f}_{WalkerIndex:.6f}" \
                   f"__zstart{self.Redshifts[RedshiftIndex]}_zend{self.Redshifts[RedshiftIndex + 1]}_" \
                   f"FLIPBOXES0_{self.BoxRes}_{self.BoxSize}Mpc_lighttravel"
        # print(filepath)
        return filepath

    def LoadBox(self, RedshiftIndex, WalkerIndex):
        if RedshiftIndex < 0 or RedshiftIndex > self.NumBoxes:
            raise ValueError(f"should be between 0 and {self.NumBoxes-1}")
        if WalkerIndex < 0 or WalkerIndex > self.WalkerSteps:
            raise ValueError(f"should be between 0 and {self.WalkerSteps - 1}")

        filepath = self.CreateFilepath(RedshiftIndex, WalkerIndex)
        
        return np.fromfile(open(filepath,'rb'), dtype = np.dtype('float32'), \
                    count = int(self.BoxRes)**3).reshape((int(self.BoxRes), int(self.BoxRes), int(self.BoxRes)))
        
    def CombineBoxes(self, WalkerIndex, NumberOfBoxes, StartIndex = 0):
        """
        Connecting all boxes between StartIndex and StarIndex + NumberOfBoxes
        """
        Box = self.LoadBox(StartIndex, WalkerIndex)
        for i in range(StartIndex + 1, StartIndex + NumberOfBoxes):
            #not sure about the axis = 0, 1, 2? It seems from 21cmFAST, it should be axis=0
            Box = np.concatenate((Box, self.LoadBox(i, WalkerIndex)), axis=0) 
        return Box

# def MiddleSlice(Box):
#     BoxDim = Box.shape
#     return Box[:, :, BoxDim[2] // 2].T

def SliceBoxNTimesXY(Box, N):
    BoxDim = Box.shape
    slices = np.zeros((2 * N, BoxDim[1], BoxDim[0]))
    for x in range(N):
        slices[x] = Box[:, x * BoxDim[1] // N, :].T
    for y in range(N):
        slices[y+N] = Box[:, :, y * BoxDim[2] // N].T

    return slices