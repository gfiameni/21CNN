import numpy as np

class Database:
    def __init__(self, Parameters, Redshifts, BoxesPath = "", ParametersPath = "", BoxType = "delta_T", BoxRes = 200, BoxSize = 300, WalkerID = 0, WalkerSteps = 10000):
        self.BoxesPath = BoxesPath
        self.ParametersPath = ParametersPath
        self.BoxType = BoxType
        self.BoxRes = BoxRes
        self.BoxSize = BoxSize
        self.WalkerID = WalkerID
        self.Redshifts = Redshifts
        self.WalkerSteps = WalkerSteps
        self.Parameters = Parameters

        self.NumBoxes = len(Redshifts) - 1

    def CreateFilepath(self, RedshiftIndex, WalkerIndex):
        filepath = f"{self.BoxesPath}/{self.BoxType}_{self.WalkerID:.6f}_{WalkerIndex:.6f}" \
                   f"__zstart{self.Redshifts[RedshiftIndex]}_zend{self.Redshifts[RedshiftIndex + 1]}_" \
                   f"FLIPBOXES0_{self.BoxRes}_{self.BoxSize}Mpc_lighttravel"
        # print(filepath)
        return filepath
    def CreateParamFilepath(self, WalkerIndex):
        filepath = f"{self.ParametersPath}/Walker_{self.WalkerID:.6f}_{WalkerIndex:.6f}.txt"
        return filepath

    def LoadBox(self, RedshiftIndex, WalkerIndex):
        if RedshiftIndex < 0 or RedshiftIndex > self.NumBoxes:
            raise ValueError(f"should be between 0 and {self.NumBoxes-1}")
        if WalkerIndex < 0 or WalkerIndex > self.WalkerSteps:
            raise ValueError(f"should be between 0 and {self.WalkerSteps - 1}")

        filepath = self.CreateFilepath(RedshiftIndex, WalkerIndex)
        
        # return np.fromfile(open(filepath,'rb'), dtype = np.dtype('float32'), \
        #             count = int(self.BoxRes)**3).reshape((int(self.BoxRes), int(self.BoxRes), int(self.BoxRes)))
        f = np.fromfile(open(filepath,'rb'), dtype = np.dtype('float32'))
        f = f.reshape((int(self.BoxRes), int(self.BoxRes), int(len(f) / self.BoxRes**2))) #I assume z is axis=2, therefore last axis is not generally dim=BoxRes
        return f
    def CombineBoxes(self, WalkerIndex, NumberOfBoxes = 1e4, StartIndex = 0):
        """
        Connecting all boxes between StartIndex and StarIndex + NumberOfBoxes
        """
        if NumberOfBoxes > self.NumBoxes:
            NumberOfBoxes = self.NumBoxes
        
        Box = self.LoadBox(StartIndex, WalkerIndex)
        # print(Box.shape)
        for i in range(StartIndex + 1, StartIndex + NumberOfBoxes):
            #not sure about the axis = 0, 1, 2? It seems from 21cmFAST, it should be axis=0
            #but from created images, axis 2 is the right one
            Box = np.concatenate((Box, self.LoadBox(i, WalkerIndex)), axis=2) 
            # print(i)
        return Box

    def WalkerAstroParams(self, WalkerIndex, ReturnType = "dict"):
        """
        Reads Astro params, saves in dict if ReturnType == "dict", else for ReturnType == "array" returns numpy.array
        """
        filepath = self.CreateParamFilepath(WalkerIndex)
        d = {}
        with open(filepath) as f:
            for line in f:
                if len(line.split()) != 2:
                    continue

                (key, val) = line.split()
                if key in self.Parameters:
                    d[key] = float(val)

        if ReturnType is "dict":
            return d
        else:
            if ReturnType is not "array":
                raise TypeError
            a = []
            for p in self.Parameters:
                a.append(d[p])
            return np.array(a)



def MiddleSlice(Box):
    BoxDim = Box.shape
    return Box[BoxDim[0] // 2, :, :]

def SliceBoxNTimesXY(Box, N):
    BoxDim = Box.shape
    slices = np.zeros((2 * N, BoxDim[1], BoxDim[2]))
    for x in range(N):
        slices[x] = Box[x * BoxDim[0] // N, :, :]
    for y in range(N):
        slices[y+N] = Box[:, y * BoxDim[1] // N, :]

    return slices

def CreateSlicedData(db, SlicesPerAxis = 5):
    """
    Creating general sliced cubes without post or preprocessing
    db == DatabaseUtils.Database object
    SlicesPerAxis -> cube is sliced in equal intervals SlicesPerAxis times in X and Y
    """
    FinalData = []
    for i in range(db.WalkerSteps):
        Box = db.CombineBoxes(i)
        BoxSlices = SliceBoxNTimesXY(Box, SlicesPerAxis)
        FinalData.append(BoxSlices)
        if i%100 == 0:
            print(i)
    return np.array(FinalData)