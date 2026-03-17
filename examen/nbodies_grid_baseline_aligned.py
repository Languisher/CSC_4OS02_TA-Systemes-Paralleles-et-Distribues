import sys

import numpy as np
import visualizer3d

G = 1.560339e-13


def generate_star_color(mass: float) -> tuple[int, int, int]:
    if mass > 5.0:
        return (150, 180, 255)
    if mass > 2.0:
        return (255, 255, 255)
    if mass >= 1.0:
        return (255, 255, 200)
    return (255, 150, 100)


def update_stars_in_grid(
    cell_start_indices: np.ndarray,
    body_indices: np.ndarray,
    cell_masses: np.ndarray,
    cell_com_positions: np.ndarray,
    masses: np.ndarray,
    positions: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
):
    del grid_max
    n_bodies = positions.shape[0]
    n_cells_total = int(np.prod(n_cells))
    cell_start_indices.fill(-1)
    cell_counts = np.zeros(shape=(n_cells_total,), dtype=np.int64)

    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for axis in range(3):
            if cell_idx[axis] >= n_cells[axis]:
                cell_idx[axis] = n_cells[axis] - 1
            elif cell_idx[axis] < 0:
                cell_idx[axis] = 0
        morse_idx = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1]
        cell_counts[morse_idx] += 1

    running_index = 0
    for i in range(n_cells_total):
        cell_start_indices[i] = running_index
        running_index += cell_counts[i]
    cell_start_indices[n_cells_total] = running_index

    current_counts = np.zeros(shape=(n_cells_total,), dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for axis in range(3):
            if cell_idx[axis] >= n_cells[axis]:
                cell_idx[axis] = n_cells[axis] - 1
            elif cell_idx[axis] < 0:
                cell_idx[axis] = 0
        morse_idx = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1]
        index_in_cell = cell_start_indices[morse_idx] + current_counts[morse_idx]
        body_indices[index_in_cell] = ibody
        current_counts[morse_idx] += 1

    cell_masses.fill(0.0)
    cell_com_positions.fill(0.0)
    for i in range(n_cells_total):
        cell_mass = 0.0
        com_position = np.zeros(3, dtype=np.float32)
        start_idx = cell_start_indices[i]
        end_idx = cell_start_indices[i + 1]
        for j in range(start_idx, end_idx):
            ibody = body_indices[j]
            mass = masses[ibody]
            cell_mass += mass
            com_position += positions[ibody] * mass
        if cell_mass > 0.0:
            com_position /= cell_mass
        cell_masses[i] = cell_mass
        cell_com_positions[i] = com_position


def compute_acceleration(
    positions: np.ndarray,
    masses: np.ndarray,
    cell_start_indices: np.ndarray,
    body_indices: np.ndarray,
    cell_masses: np.ndarray,
    cell_com_positions: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
):
    del grid_max
    n_bodies = positions.shape[0]
    a = np.zeros_like(positions)
    for ibody in range(n_bodies):
        pos = positions[ibody]
        cell_idx = np.floor((pos - grid_min) / cell_size).astype(np.int64)
        for axis in range(3):
            if cell_idx[axis] >= n_cells[axis]:
                cell_idx[axis] = n_cells[axis] - 1
            elif cell_idx[axis] < 0:
                cell_idx[axis] = 0
        for ix in range(n_cells[0]):
            for iy in range(n_cells[1]):
                for iz in range(n_cells[2]):
                    morse_idx = ix + iy * n_cells[0] + iz * n_cells[0] * n_cells[1]
                    if (abs(ix - cell_idx[0]) > 2) or (abs(iy - cell_idx[1]) > 2) or (abs(iz - cell_idx[2]) > 2):
                        cell_mass = cell_masses[morse_idx]
                        if cell_mass <= 0.0:
                            continue
                        direction = cell_com_positions[morse_idx] - pos
                        distance = np.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
                        if distance > 1.0e-10:
                            inv_dist3 = 1.0 / (distance ** 3)
                            a[ibody, :] += G * direction[:] * inv_dist3 * cell_mass
                    else:
                        start_idx = cell_start_indices[morse_idx]
                        end_idx = cell_start_indices[morse_idx + 1]
                        for j in range(start_idx, end_idx):
                            jbody = body_indices[j]
                            if jbody == ibody:
                                continue
                            direction = positions[jbody] - pos
                            distance = np.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
                            if distance > 1.0e-10:
                                inv_dist3 = 1.0 / (distance ** 3)
                                a[ibody, :] += G * direction[:] * inv_dist3 * masses[jbody]
    return a


class SpatialGrid:
    def __init__(self, positions: np.ndarray, nb_cells_per_dim: tuple[int, int, int]):
        self.min_bounds = np.min(positions, axis=0) - 1.0e-6
        self.max_bounds = np.max(positions, axis=0) + 1.0e-6
        self.n_cells = np.array(nb_cells_per_dim)
        self.cell_size = (self.max_bounds - self.min_bounds) / self.n_cells
        self.cell_start_indices = np.full(np.prod(self.n_cells) + 1, -1, dtype=np.int64)
        self.body_indices = np.empty(shape=(positions.shape[0],), dtype=np.int64)
        self.cell_masses = np.zeros(shape=(np.prod(self.n_cells),), dtype=np.float32)
        self.cell_com_positions = np.zeros(shape=(np.prod(self.n_cells), 3), dtype=np.float32)

    def update(self, positions: np.ndarray, masses: np.ndarray):
        update_stars_in_grid(
            self.cell_start_indices,
            self.body_indices,
            self.cell_masses,
            self.cell_com_positions,
            masses,
            positions,
            self.min_bounds,
            self.max_bounds,
            self.cell_size,
            self.n_cells,
        )


class NBodySystem:
    def __init__(self, filename, ncells_per_dir: tuple[int, int, int] = (10, 10, 10)):
        positions = []
        velocities = []
        masses = []

        self.max_mass = 0.0
        self.box = np.array([[-1.0e-6, -1.0e-6, -1.0e-6], [1.0e-6, 1.0e-6, 1.0e-6]], dtype=np.float64)
        with open(filename, "r", encoding="utf-8") as fich:
            line = fich.readline()
            while line:
                data = line.split()
                masses.append(float(data[0]))
                positions.append([float(data[1]), float(data[2]), float(data[3])])
                velocities.append([float(data[4]), float(data[5]), float(data[6])])
                self.max_mass = max(self.max_mass, masses[-1])

                for i in range(3):
                    self.box[0][i] = min(self.box[0][i], positions[-1][i] - 1.0e-6)
                    self.box[1][i] = max(self.box[1][i], positions[-1][i] + 1.0e-6)

                line = fich.readline()

        self.positions = np.array(positions, dtype=np.float32)
        self.velocities = np.array(velocities, dtype=np.float32)
        self.masses = np.array(masses, dtype=np.float32)
        self.colors = [generate_star_color(m) for m in masses]
        self.grid = SpatialGrid(self.positions, ncells_per_dir)
        self.grid.update(self.positions, self.masses)

    def update_positions(self, dt):
        a = compute_acceleration(
            self.positions,
            self.masses,
            self.grid.cell_start_indices,
            self.grid.body_indices,
            self.grid.cell_masses,
            self.grid.cell_com_positions,
            self.grid.min_bounds,
            self.grid.max_bounds,
            self.grid.cell_size,
            self.grid.n_cells,
        )
        self.positions += self.velocities * dt + 0.5 * a * dt * dt
        self.grid.update(self.positions, self.masses)
        a_new = compute_acceleration(
            self.positions,
            self.masses,
            self.grid.cell_start_indices,
            self.grid.body_indices,
            self.grid.cell_masses,
            self.grid.cell_com_positions,
            self.grid.min_bounds,
            self.grid.max_bounds,
            self.grid.cell_size,
            self.grid.n_cells,
        )
        self.velocities += 0.5 * (a + a_new) * dt


system: NBodySystem


def update_positions(dt: float):
    global system
    system.update_positions(dt)
    return system.positions


def run_simulation(filename, geometry=(800, 600), ncells_per_dir: tuple[int, int, int] = (10, 10, 10), dt=0.001):
    global system
    system = NBodySystem(filename, ncells_per_dir=ncells_per_dir)
    pos = system.positions
    col = system.colors
    intensity = np.clip(system.masses / system.max_mass, 0.5, 1.0)
    visu = visualizer3d.Visualizer3D(
        pos,
        col,
        intensity,
        [[system.box[0][0], system.box[1][0]], [system.box[0][1], system.box[1][1]], [system.box[0][2], system.box[1][2]]],
    )
    visu.run(updater=update_positions, dt=dt)


def parse_args(argv):
    filename = "data/galaxy_1000"
    dt = 0.001
    n_cells_per_dir = (20, 20, 1)
    if len(argv) > 1:
        filename = argv[1]
    if len(argv) > 2:
        dt = float(argv[2])
    if len(argv) > 5:
        n_cells_per_dir = (int(argv[3]), int(argv[4]), int(argv[5]))
    return filename, dt, n_cells_per_dir


if __name__ == "__main__":
    filename, dt, n_cells_per_dir = parse_args(sys.argv)
    print(f"Simulation baseline alignée de {filename} avec dt = {dt} et grille {n_cells_per_dir}")
    run_simulation(filename, ncells_per_dir=n_cells_per_dir, dt=dt)
