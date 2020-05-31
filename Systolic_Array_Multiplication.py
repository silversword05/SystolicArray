from queue import Queue

import numpy as np


# https://www.youtube.com/watch?v=cmy7LBaWuZ8 for visualization
class SystolicCell:
    def __init__(self, row: float, col: float):
        # position of the cell
        self.pos_row: float = row
        self.pos_col: float = col
        # current sum at a clock cycle
        self.current_sum: float = 0
        # state values to be passed on in next iteration
        self.state_a: float = 0
        self.state_b: float = 0
        # To hold values when a particular clock cycle is occurring
        self.state_a_stage: float = 0
        self.state_b_stage: float = 0

    def perform_clock_cycle(self, new_a: float, new_b: float):
        # calculating the sum
        self.current_sum += new_a * new_b
        # putting new values at stage, to be updated after one clock occurs
        self.state_a_stage = new_a
        self.state_b_stage = new_b

    def update_states(self):
        # updating the stage values after the clock cycle completes
        self.state_a = self.state_a_stage
        self.state_b = self.state_b_stage


class QueueMatrixForm:
    def __init__(self, arr: np.ndarray, transpose=False):
        # storing basic array details
        if transpose:
            arr = arr.transpose()
        self.array_rows = len(arr)
        self.array_columns = len(arr[0])
        self.array_queues = []
        # converting each row to a queue
        for i, row in enumerate(arr):
            row_invert = row[::-1]
            q = Queue()
            for p in range(i):
                q.put(0)
            for x in row_invert:
                q.put(x)
            self.array_queues.append(q)


class SystolicArray:
    def __init__(self, a_queues: QueueMatrixForm, b_queues: QueueMatrixForm):
        # calculating the final result array dimension
        self.res_row_cnt = a_queues.array_rows
        self.res_col_cnt = b_queues.array_rows
        # getting the input in form of queues
        self.a_queues = a_queues
        self.b_queues = b_queues
        # creating the systolic cells to form the systolic array
        self.sys_arr = []
        for row_pos in range(self.res_row_cnt):
            sys_row = [SystolicCell(row_pos, col_pos) for col_pos in range(self.res_col_cnt)]
            self.sys_arr.append(sys_row)

    def iterate_one_cycle(self):
        for row_pos in range(self.res_row_cnt):
            for col_pos in range(self.res_col_cnt):
                # arranging data from the previous cell or getting it from the queue. 0 for no queue value
                if row_pos == 0 and col_pos != 0:
                    data_b = self.b_queues.array_queues[col_pos].get() if not self.b_queues.array_queues[col_pos].empty() else 0
                    data_a = self.sys_arr[row_pos][col_pos - 1].state_a
                elif col_pos == 0 and row_pos != 0:
                    data_a = self.a_queues.array_queues[row_pos].get() if not self.a_queues.array_queues[row_pos].empty() else 0
                    data_b = self.sys_arr[row_pos - 1][col_pos].state_b
                elif row_pos == 0 and col_pos == 0:
                    data_a = self.a_queues.array_queues[row_pos].get() if not self.a_queues.array_queues[row_pos].empty() else 0
                    data_b = self.b_queues.array_queues[col_pos].get() if not self.b_queues.array_queues[col_pos].empty() else 0
                else:
                    data_a = self.sys_arr[row_pos][col_pos - 1].state_a
                    data_b = self.sys_arr[row_pos - 1][col_pos].state_b
                # print(row_pos, col_pos, data_a, data_b)
                self.sys_arr[row_pos][col_pos].perform_clock_cycle(data_a, data_b)
        # updating the new states
        for row_pos in range(self.res_row_cnt):
            for col_pos in range(self.res_col_cnt):
                self.sys_arr[row_pos][col_pos].update_states()

    def return_array(self):
        # Converting Systolic array to numpy array
        res = np.zeros(shape=(self.res_row_cnt, self.res_col_cnt), dtype=float)
        for row_pos in range(self.res_row_cnt):
            for col_pos in range(self.res_col_cnt):
                res[row_pos][col_pos] = self.sys_arr[row_pos][col_pos].current_sum
        return res


arr1 = np.arange(15).reshape((5, 3))
arr2 = np.arange(6).reshape((3, 2))
arr_test1 = QueueMatrixForm(arr1)
arr_test2 = QueueMatrixForm(arr2, transpose=True)  # This is necessary as the other array queues are formed in transposed manner
obj = SystolicArray(arr_test1, arr_test2)
iter_no = max(arr_test1.array_rows + arr_test1.array_columns, arr_test2.array_rows + arr_test2.array_columns)
for i in range(arr_test1.array_columns + arr_test1.array_rows):
    obj.iterate_one_cycle()
    # print(obj.return_array())
print(obj.return_array())
print((obj.return_array() == np.matmul(arr1, arr2)).all())
