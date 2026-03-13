import numpy as np
from pathfinding.core.grid import Grid as PFGrid
from pathfinding.finder.a_star import AStarFinder

class GridManager:
    def __init__(self, width=100, height=100, scale=0.1):
        self.width = width
        self.height = height
        self.scale = scale  # 0.1 -> 1 sim unit = 10 grid squares
        self.grid = np.ones((width, height), dtype=int)  # 1 = walkable, 0 = blocked

    def world_to_grid(self, pos):
        x, y = int(pos[0] / self.scale), int(pos[1] / self.scale)
        return max(0, min(x, self.width - 1)), max(0, min(y, self.height - 1))

    def grid_to_world(self, x, y):
        return x * self.scale + self.scale / 2, y * self.scale + self.scale / 2

    def update_from_environment(self, environment):
        self.grid[:, :] = 1  # reset
        for name, obj in environment.objects.items():
            if obj["type"] in {"rect", "circle", "blocked"} and not obj.get("passable", False):
                self._block_object(obj)

    def _block_object(self, obj):
        if obj["type"] == "rect":
            x, y = obj["position"]
            w, h = obj["size"]
            x0, y0 = self.world_to_grid((x, y))
            x1, y1 = self.world_to_grid((x + w, y + h))
        elif obj["type"] == "circle":
            cx, cy = obj["position"]
            r = obj["radius"]
            x0, y0 = self.world_to_grid((cx - r, cy - r))
            x1, y1 = self.world_to_grid((cx + r, cy + r))
        elif obj["type"] == "blocked":
            (x0f, y0f), (x1f, y1f) = obj["corners"]
            x0, y0 = self.world_to_grid((x0f, y0f))
            x1, y1 = self.world_to_grid((x1f, y1f))

        for gx in range(min(x0, x1), max(x0, x1) + 1):
            for gy in range(min(y0, y1), max(y0, y1) + 1):
                if 0 <= gx < self.width and 0 <= gy < self.height:
                    self.grid[gx, gy] = 0  # blocked

    def find_path(self, start, end):
        grid_obj = PFGrid(matrix=self.grid.T.tolist())  # Transpose to (y,x)
        sx, sy = self.world_to_grid(start)
        ex, ey = self.world_to_grid(end)
        start_node = grid_obj.node(sx, sy)
        end_node = grid_obj.node(ex, ey)
        finder = AStarFinder()
        path, _ = finder.find_path(start_node, end_node, grid_obj)
        return [self.grid_to_world(x, y) for x, y in path]
