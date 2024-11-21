#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define ALIVE 1
#define DEAD 0

// Kernel to simulate one generation
__global__ void game_of_life(const int *current, int *next, int rows, int cols) {
    // Get the row and column index for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread is within bounds
    if (row >= rows || col >= cols) return;

    // Count live neighbors (with wrapping for toroidal grid)
    int live_neighbors = 0;
    for (int dr = -1; dr <= 1; dr++) {
        for (int dc = -1; dc <= 1; dc++) {
            if (dr == 0 && dc == 0) continue; // Skip the cell itself
            int neighbor_row = (row + dr + rows) % rows; // Wrap row
            int neighbor_col = (col + dc + cols) % cols; // Wrap col
            live_neighbors += current[neighbor_row * cols + neighbor_col];
        }
    }

    // Compute the next state of the cell
    int idx = row * cols + col;
    if (current[idx] == ALIVE && (live_neighbors < 2 || live_neighbors > 3)) {
        next[idx] = DEAD; // Underpopulation or Overpopulation
    } else if (current[idx] == DEAD && live_neighbors == 3) {
        next[idx] = ALIVE; // Reproduction
    } else {
        next[idx] = current[idx]; // Survival
    }
}

int main() {
    const int rows = 32, cols = 32; // Size of the grid
    const int generations = 100;   // Number of generations
    size_t size = rows * cols * sizeof(int);

    // Host grids
    std::vector<int> h_current(rows * cols, DEAD);
    std::vector<int> h_next(rows * cols, DEAD);

    // Initialize the grid with some pattern (e.g., Glider)
    h_current[1 * cols + 2] = ALIVE;
    h_current[2 * cols + 3] = ALIVE;
    h_current[3 * cols + 1] = ALIVE;
    h_current[3 * cols + 2] = ALIVE;
    h_current[3 * cols + 3] = ALIVE;

    // Allocate device memory
    int *d_current, *d_next;
    cudaMalloc(&d_current, size);
    cudaMalloc(&d_next, size);

    // Copy initial grid to the device
    cudaMemcpy(d_current, h_current.data(), size, cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

    // Simulate generations
    for (int gen = 0; gen < generations; gen++) {
        game_of_life<<<gridDim, blockDim>>>(d_current, d_next, rows, cols);
        cudaDeviceSynchronize();

        // Swap the grids
        std::swap(d_current, d_next);
    }

    // Copy final grid back to the host
    cudaMemcpy(h_current.data(), d_current, size, cudaMemcpyDeviceToHost);

    // Print the final grid
    std::cout << "Final Grid:" << std::endl;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            std::cout << (h_current[r * cols + c] ? "#" : ".") << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_current);
    cudaFree(d_next);

    return 0;
}
