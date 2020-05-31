import numpy as np


class SystolicWVCell:
    def __init__(self, row: float, col: float, weight: float):
        # position of the cell
        self.pos_row: float = row
        self.pos_col: float = col
        # current sum at a clock cycle
        self.current_sum: float = 0
        self.weight_value: float = weight
        # state values to be passed on in next iteration
        self.state_sum: float = 0
        self.state_activation: float = 0
        # To hold values when a particular clock cycle is occurring
        self.__state_sum_stage: float = 0
        self.__state_activation_stage: float = 0

    def perform_clock_cycle(self, new_activation: float, new_sum: float):
        # calculating the sum
        self.current_sum = new_activation * self.weight_value + new_sum
        # putting new values at stage, to be updated after one clock occurs
        self.__state_sum_stage = self.current_sum
        self.__state_activation_stage = new_activation

    def update_states(self):
        # updating the stage values after the clock cycle completes
        self.state_sum = self.__state_sum_stage
        self.state_activation = self.__state_activation_stage


class SystolicWVArray:
    def __init__(self, weight_arr: np.ndarray, activation_arr: np.ndarray):
        self.weight_arr = weight_arr  # weight array 2D
        self.activation_arr = activation_arr  # activation array 1D
        if weight_arr.shape[0] != len(activation_arr):
            raise Exception(" Shape mismatch error. Check row dimension of weight and number of activations")
        # creating the systolic cells to form the systolic array
        self.sys_arr = []
        for row in range(weight_arr.shape[0]):
            sys_row = [SystolicWVCell(row, col, self.weight_arr[row][col]) for col in range(weight_arr.shape[1])]
            self.sys_arr.append(sys_row)
        self.__clock: int = 0 # maintaining the clock
        self.res = np.zeros(shape=self.weight_arr.shape[1], dtype=float)

    def iterate_one_cycle(self):
        for row in range(self.weight_arr.shape[0]):
            for col in range(self.weight_arr.shape[1]):
                # getting data for the cells. Activations are entered only when clock matches with the row number
                if row == 0 and col == 0:
                    data_sum = 0
                    data_activation = self.activation_arr[row] if self.__clock == row else 0
                elif row == 0 and col != 0:
                    data_sum = 0
                    data_activation = self.sys_arr[row][col-1].state_activation
                elif row != 0 and col == 0:
                    data_sum = self.sys_arr[row-1][col].state_sum
                    data_activation = self.activation_arr[row] if self.__clock == row else 0
                else:
                    data_sum = self.sys_arr[row-1][col].state_sum
                    data_activation = self.sys_arr[row][col-1].state_activation
                self.sys_arr[row][col].perform_clock_cycle(data_activation, data_sum)
        # updating new states
        for row in range(self.weight_arr.shape[0]):
            for col in range(self.weight_arr.shape[1]):
                self.sys_arr[row][col].update_states()
        self.__clock += 1

    def get_result(self):
        last_row_sys_arr = self.sys_arr[-1] # the last row stores the final result
        for x in range(self.weight_arr.shape[1]):
            self.res[x] += last_row_sys_arr[x].state_sum

    def get_systolic_array(self):
        arr = np.zeros(shape=self.weight_arr.shape, dtype=float)
        for row in range(self.weight_arr.shape[0]):
            for col in range(self.weight_arr.shape[1]):
                arr[row][col] = self.sys_arr[row][col].state_sum
        return arr


activation = np.array([1,2,3])
weight = np.array([[1,2],[4,5],[7,8]])

obj = SystolicWVArray(weight, activation)
for clock in range(sum(weight.shape)):
    obj.iterate_one_cycle()
    obj.get_result()

print(obj.res)
print((obj.res == np.matmul(activation, weight)).all())