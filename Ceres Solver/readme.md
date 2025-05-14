
#  Ceres Bundle Adjustment API

This part of the project is a lightweight RESTful API server for performing bundle adjustment using the **Ceres Solver**, **Crow** (a C++ web framework), and other essential C++ libraries like **Eigen** and **Boost**.

It provides a `/bundle_adjust` POST endpoint that takes camera parameters, 3D points, 2D observations, and camera intrinsics, then runs optimization and returns refined parameters.

---

## Dependencies

Make sure the following dependencies are installed on your system:

### Required Libraries

- Ceres Solver  
- Eigen3  
- Boost (>= 1.83) — with `system` and `filesystem` components  
- OpenSSL  
- CMake >= 3.10  
- A C++ compiler that supports C++17 or higher (e.g., `g++` or `clang++`)

On Ubuntu, you can install the required dependencies via:

```bash
sudo apt-get update
sudo apt-get install cmake libceres-dev libeigen3-dev libboost-all-dev libssl-dev
```

---
##  Project Structure

bash


├── CMakeLists.txt

├── main.cpp

└── build/
## Build Instructions

1. **Clean previous builds** (optional but recommended):

   ```bash
   cd build
   rm -rf *
   ```

2. **Configure with CMake:**

   ```bash
   cmake ..
   ```

3. **Build the executable:**

   ```bash
   make
   ```

4. **Run the server:**

   ```bash
   ./reconstruction
   ```

   You should see:

   ```
   Starting Ceres optimization server on port 8080...
   ```

   The server will now be listening at:  
   `http://localhost:8080`

---

##  API Endpoints

### `POST /bundle_adjust`

Run bundle adjustment optimization.

#### Request Body (JSON)

```json
{
  "observations": [150.0, 200.0, 152.0, 198.5],
  "camera_params": [1.0, 1.0, ..., 1.0],  // 12 values
  "points_3d": [1.0, 2.0, 3.0, ..., 6.0], // Multiple 3D points
  "intrinsics": [800.0, 800.0, 320.0, 240.0]
}
```

#### Response

```json
{
  "optimized_camera": [...],
  "optimized_points": [...],
  "summary": {
    "brief_report": "...",
    "total_time": 0.0123,
    "initial_cost": 123.456,
    "final_cost": 78.910,
    "iterations": 25
  }
}
```

---
##  Testing Example (Python)

Here's a basic example using Python and `requests` or you can implement our sfm project to use it on the real datasets:

```python
import requests
import json

payload = {
    "observations": [150.0, 200.0, 152.0, 198.5],
    "camera_params": [1.0]*12,
    "points_3d": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "intrinsics": [800.0, 800.0, 320.0, 240.0]
}

res = requests.post("http://localhost:8080/bundle_adjust", json=payload)
print(res.json())
```

---

## Notes

- The API assumes valid input shapes and values; malformed requests may result in undefined behavior or server errors



##  Contact

If you have any queries, issues, or need assistance, feel free to reach out to the team:

- [@malikdanialahmed](https://github.com/malikdanialahmed)  
  **Danial Ahmed** — Collaborator
- [@hassaanahmed04](https://github.com/hassaanahmed04)  
  **Hassaan Ahmed** — Collaborator
- [@Ureed-Hussain](https://github.com/Ureed-Hussain)  
  **Muhammad Ureed Hussain** — Collaborator

