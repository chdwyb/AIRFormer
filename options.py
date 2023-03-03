
class Options():
    def __init__(self):
        super().__init__()

        self.Warmup_Epochs = 3

        self.Seed = 1234
        self.Epoch = 200
        self.Learning_Rate = 2e-4
        self.Batch_Size_Train = 1
        self.Batch_Size_Val = 1
        self.Patch_Size_Train = 256
        self.Patch_Size_Val = 256
        # './AIR40K/train/input/'
        self.Input_Path_Train = './AIR40K/test/Rain100H/input/'
        self.Target_Path_Train = './AIR40K/test/Rain100H/target/'

        self.Input_Path_Val = './AIR40K/test/Rain100H/input/'
        self.Target_Path_Val = './AIR40K/test/Rain100H/target/'

        flag = 14
        dict = {
            0: 'Rain100H',
            1: 'Rain100L',
            2: 'Rain100',
            3: 'Rain1200',
            4: 'Rain2800',
            5: 'Rain12',
            6: 'RainDA',
            7: 'RainDB',
            8: 'HazeIn',
            9: 'HazeOut',
            10: 'Haze10',
            11: 'SnowL',
            12: 'SnowM',
            13: 'SnowS',
            14: 'Snow2000',
        }
        choice = dict[flag]
        self.Input_Path_Test = './AIR40K/test/'+choice+'/input/'
        self.Target_Path_Test = './AIR40K/test/'+choice+'/target/'
        self.Result_Path_Test = './AIR40K/test/'+choice+'/result/'

        self.Num_Works = 4
        self.CUDA_USE = True
        self.Pre_Train = True