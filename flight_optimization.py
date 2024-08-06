import pandas as pd
import numpy as np
from scipy.optimize import linprog
from tqdm import tqdm

# 读取CSV文件
file_path = 'data.csv'
data = pd.read_csv(file_path)

# 当前时间
current_time = 1000

# 延误的飞机
delayed_airplanes = list(range(1, 13))

# 初始化新的起飞时间列
data['new_depTime'] = data['depTime']

# 固定的航班
fixed_flights = [0, 14, 29, 42, 56, 71, 87, 101, 114, 129, 142, 154]

# 将固定航班的起飞时间和降落时间设为固定
data.loc[data['series'].isin(fixed_flights), 'new_depTime'] = data['depTime']

# 设置迭代轮数上限
max_iterations = 500


# 定义计算延误费用的函数
def calculate_delay_fee(dep_time, new_dep_time, delay_fee):
    if new_dep_time > dep_time:
        return (new_dep_time - dep_time) * delay_fee
    return 0


# 更新航班时间的函数，确保时空逻辑约束
def update_flight_times(data):
    for airplane in data['airplane'].unique():
        airplane_flights = data[data['airplane'] == airplane].sort_values(by='depTime')
        last_arrival_time = 0
        last_arrival_airport = None
        for index, flight in airplane_flights.iterrows():
            if flight['series'] not in fixed_flights:
                dep_time = max(flight['depTime'], last_arrival_time)
                # 确保飞机从上一个航班的降落机场起飞
                if last_arrival_airport is not None and flight['depPort'] != last_arrival_airport:
                    dep_time = last_arrival_time + 1  # 加入转场时间
                data.at[index, 'new_depTime'] = dep_time
                last_arrival_time = dep_time + flight['time']
                last_arrival_airport = flight['arrPort']
            else:
                last_arrival_time = flight['arrTime']
                last_arrival_airport = flight['arrPort']


# 主问题求解的函数
def solve_master_problem(data):
    c = []
    for index, row in data.iterrows():
        if row['series'] in fixed_flights:
            c.append(0)
        else:
            delay_cost = calculate_delay_fee(row['depTime'], data.at[index, 'new_depTime'], row['delay_fee'])
            cancellation_cost = row['cancellation_fee']
            c.append(min(delay_cost, cancellation_cost))

    A_eq = np.zeros((len(data), len(data)))
    for i, row in data.iterrows():
        A_eq[i, i] = 1

    b_eq = np.ones(len(data))

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method='highs')

    for i, row in data.iterrows():
        if row['series'] not in fixed_flights:
            if res.x[i] < 0.5:
                data.at[i, 'new_depTime'] = np.nan
            else:
                data.at[i, 'new_depTime'] = max(row['depTime'], current_time)


# 子问题求解的函数
def solve_sub_problem(data):
    new_routes = []
    for i, row in data.iterrows():
        if row['series'] not in fixed_flights:
            if data.at[i, 'new_depTime'] >= row['depTime'] and data.at[i, 'new_depTime'] <= current_time + row['time']:
                new_route = {
                    'series'           : row['series'],
                    'airplane'         : row['airplane'],
                    'depPort'          : row['depPort'],
                    'arrPort'          : row['arrPort'],
                    'depTime'          : data.at[i, 'new_depTime'],
                    'arrTime'          : data.at[i, 'new_depTime'] + row['time'],
                    'delay_cost'       : calculate_delay_fee(row['depTime'], data.at[i, 'new_depTime'],
                                                             row['delay_fee']),
                    'cancellation_cost': row['cancellation_fee']
                }
                new_routes.append(new_route)
    return new_routes


# 初始化并运行列生成算法
def generate_initial_routes(data):
    routes = []
    for index, row in data.iterrows():
        if row['airplane'] in delayed_airplanes and row['series'] not in fixed_flights:
            route = {
                'series'           : row['series'],
                'airplane'         : row['airplane'],
                'depPort'          : row['depPort'],
                'arrPort'          : row['arrPort'],
                'depTime'          : current_time,
                'arrTime'          : current_time + row['time'],
                'delay_cost'       : calculate_delay_fee(row['depTime'], current_time, row['delay_fee']),
                'cancellation_cost': row['cancellation_fee']
            }
            routes.append(route)
    return routes


def column_generation(data):
    routes = generate_initial_routes(data)
    for iteration in tqdm(range(max_iterations), desc="进度"):
        solve_master_problem(data)
        update_flight_times(data)
        new_routes = solve_sub_problem(data)
        if not new_routes:
            break
        routes.extend(new_routes)
    return routes


column_generation(data)

total_cost = 0
for index, row in data.iterrows():
    if row['series'] not in fixed_flights:
        dep_time = row['depTime']
        new_dep_time = row['new_depTime']
        delay_fee = row['delay_fee']
        cancellation_fee = row['cancellation_fee']

        delay_cost = calculate_delay_fee(dep_time, new_dep_time, delay_fee)
        if cancellation_fee < delay_cost:
            data.at[index, 'new_depTime'] = np.nan
            total_cost += cancellation_fee
        else:
            total_cost += delay_cost

output_file_path = 'new_data.csv'
data.to_csv(output_file_path, index=False)

print("最小费用和:", total_cost)
