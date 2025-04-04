import math

# Grid NMS 改善為O(N)
def grid_NMS(all_keypoints, cell_size):
    grid = {}

    for x, y in all_keypoints:
        cell_x = x // cell_size
        cell_y = y // cell_size
        cell_coord = (cell_x, cell_y)

        if cell_coord not in grid:
            grid[cell_coord] = (x, y)  # 該區域第一次出現 → 存入

    final_keypoints = list(grid.values())
    return final_keypoints

"""" O(N ** 2) -> 角點太會太慢
def NMS(all_keypoints, r):
    final_keypoints = set()

    for x, y in all_keypoints:
        keep = True 

        for fx, fy in final_keypoints:
            distance = math.sqrt((x - fx) ** 2 + (y - fy) ** 2)
            if distance <= r:
                keep = False
                break  # 太近的點，不保留

        if keep:
            final_keypoints.add((x, y))

    return final_keypoints
"""