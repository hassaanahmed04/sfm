
# Structure from Motion (SfM)
> *"A single image may capture a moment—but many together can reconstruct a world."*

Whether you're capturing a statue, surveying a landscape, or reimagining historical ruins, SfM is the bridge between snapshots and spatial understanding. This document is your guide through that journey—from raw images to precise reconstructions—explained with code, mathematics, and visual results.

---
## 1. Introduction

Structure from Motion (SfM) is a computer vision and photogrammetry technique that reconstructs the 3D structure of a scene and determines camera poses from a collection of 2D images. By identifying common features and solving mathematical equations, it transforms a set of images into a cohesive 3D representation, applicable in fields like 3D modeling, augmented reality, and historical landscape reconstruction. [Read more here](https://medium.com/@sepideh.92sh/unveiling-the-magic-of-3d-reconstruction-a-journey-through-structure-from-motion-sfm-in-c-d66bc1d01a96).

![Duck-min](https://github.com/user-attachments/assets/0d409065-7341-4596-9005-a83401702a41)


## SFM Pipeline Overview

The flowchart shows the SfM pipeline we followed. In the final step, only one method was used at a time for reconstruction:

- Ceres Solver [API](https://github.com/hassaanahmed04/sfm/tree/main/Ceres%20Solver)
- Bundle Adjustment (TRF, LM, Dogbox)  
- Sequential Triangulation-based Reconstruction
<img src="https://github.com/user-attachments/assets/54b0a477-e2c6-434c-a276-bb401b541968"  align="center" ><br>
## Datasets

In this project, we first implemented all the steps of Structure-from-Motion (SfM) on a pre-calibrated dataset, and later extended the pipeline to work with our own custom dataset, which involved camera calibration and careful data collection.

### Pre-Calibrated Dataset

The dataset includes 55 high-resolution images of the Gustav II Adolf statue. We also experimented with other datasets available through the provided [link](https://www.maths.lth.se/matematiklth/personal/calle/dataset/dataset.html).

## Feature Detection and Matching

In our SfM pipeline, we evaluated three feature detection and matching combinations:

- SIFT with BFMatcher (L2-norm, KNN)  
- AKAZE with Hamming distance  
- ORB with Hamming distance  

These choices were guided by best practices from [OpenCV](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html) and a research paper: [A Blender plug-in for comparing Structure from Motion pipelines](https://www.researchgate.net/publication/329751459_A_Blender_plug-in_for_comparing_Structure_from_Motion_pipelines).

<table>
  <tr>
    <td><img src="./images/sift_bf.png" width="300"/></td>
    <td><img src="./images/akaze_bf.png" width="300"/></td>
    <td><img src="./images/orb_bf.png" width="300"/></td>
  </tr>
  <tr>
    <td align="center"><strong>SIFT + BFMatcher</strong></td>
    <td align="center"><strong>AKAZE + BFMatcher</strong></td>
    <td align="center"><strong>ORB + BFMatcher</strong></td>
  </tr>
</table>

## Feature Matching Results

Based on the results, **SIFT combined with BFMatcher and KNN** yielded the most reliable feature correspondences, making it the preferred choice for our pipeline.

## Mathematical Explanation with Reference

### Initial Reconstruction

#### 1) Camera Pose Initialization
We fix the first camera as the origin using an identity pose:

$$
P_0 = K \cdot [I \mid 0]
$$

This serves as the reference frame for triangulation. Arrays for camera poses and 3D points are also initialized.

#### 2) Feature Detection and Matching

The first two images are loaded and downscaled. Features are extracted and matched using the `features_extraction()` function, which internally uses **SIFT + BFMatcher + KNN** to return store filtered features.

#### 3) Essential Matrix Estimation

We compute the **Essential Matrix** using:

```python
cv2.findEssentialMat()
```
#### Fundamental Matrix (Optional)

Although we skip explicit computation of the **Fundamental Matrix**, it can be obtained using the relation between the **Essential Matrix** and the **camera intrinsic matrix**:

$$
F = K^{-T} \cdot E \cdot K^{-1}
$$

Where:
- \( E \) is the Essential Matrix  
- \( K \) is the camera intrinsic matrix

We validate \( E \) by ensuring it has **rank 2** and \( \det(E) = 0 \).

---

## 4) Pose Recovery

We use `cv2.recoverPose()` to extract **rotation \( R \)** and **translation \( t \)** from the Essential Matrix \( E \). This function:
- Performs **Singular Value Decomposition (SVD)**
- Applies a **chirality check** to ensure triangulated 3D points lie in front of both cameras

---

## 5) Second Camera Pose Computation

The second camera’s pose is computed relative to the first by applying the recovered \( R \) and \( t \).  
The final **3×4 projection matrix** is computed as:

$$
P = K \cdot [R \mid t]
$$

Where \( K \) is the intrinsic matrix.

---

## 6) Triangulation

We use `cv2.triangulatePoints()` to reconstruct 3D points from the first and second camera projections along with their matching feature points.  
The output is in **homogeneous coordinates**, which we convert to **Euclidean** coordinates.
<img src="./images/triangulation.png"  align="center" ><br>

---

## Incremental Camera Registration (PnP)

After recovering the initial pose, we incrementally add new images by matching features with the previous frame.  
We estimate each new camera pose using:

```python
cv2.solvePnPRansac()
```

This solves the **Perspective-n-Point (PnP)** problem using RANSAC, robust to outliers, and optimizes via the **Levenberg-Marquardt algorithm** to minimize reprojection error.[link](https://medium.com/@abdulhaq.ah/solvepnpransac-and-optimization-47a0683227b1)

The `solve_pnp_prob()` function encapsulates this process and returns the camera's **rotation** and **translation**.

---

## Rotation Vector to Matrix Conversion

After pose estimation, the **rotation vector** is converted to a rotation matrix using:

```python
cv2.Rodrigues()
```

Only **inlier correspondences** are retained for accurate triangulation and further optimization.

---

## Post-Pose Estimation: Triangulation and Optimization

After estimating each camera pose:
1. New 3D points are triangulated
2. Reprojection error is calculated
3. **Bundle Adjustment** is applied to refine:
   - Camera poses  
   - 3D point positions

---

## Sequential Triangulation-Based Reconstruction

Images are added incrementally. For each new frame, the following steps are executed:
- Establish feature correspondences  
- Estimate pose using `solvePnPRansac`  
- Triangulate 3D points  
- Compute reprojection error for validation

Applied to the **Gustav II Adolf statue** dataset, this method resulted in:
- A **dense 3D point cloud** with **16,367 points**
- Processing time for **55 images: 44 seconds**
- However, high reprojection error indicated the need for further optimization
<table>
  <tr>
    <td><img src="./images/sequential_rep.png" width="450"/></td>
    <td><img src="./images/sequential_3d.png" width="450"/></td>
  </tr>
</table>
**Bundle Adjustment** was applied to refine both camera parameters and the structure, significantly improving reconstruction accuracy.

## Bundle Adjustment

After initial 3D reconstruction, reprojection errors can be high.  
To refine both **3D points** and **camera parameters**, we apply **Bundle Adjustment (BA)**—a nonlinear least-squares optimization that minimizes reprojection error across all views.

### Mathematical Formulation

Minimize the sum of Euclidean distances between observed 2D points \( x_{ij} \) and their predicted projections \( Q(a_j, b_i) \), across all visible points \( v_{ij} = 1 \):

$$
\min_{a_j, b_i} \sum_{i,j} v_{ij} \cdot \| x_{ij} - Q(a_j, b_i) \|^2
$$

Where:
- \( x_{ij} \): observed 2D point
- \( Q(a_j, b_i) \): projected 2D point from 3D point \( b_i \) and camera parameters \( a_j \)
- \( v_{ij} \): visibility flag (1 if visible)

---

### Our Implementation

We implemented **Bundle Adjustment** using `SciPy`’s `least_squares` function with three solvers:

- **Trust Region Reflective (TRF)**
- **Dogbox**
- **Levenberg-Marquardt (LM)**

---
```python
variables
method='trf'        # Can change methods like trf, or dogbox
loss='huber'        # Robust to outliers like Linear, huber, soft_l1
f_scale=1.0
ftol=1e-6
xtol=1e-6
gtol=err_thresh
max_nfev=200
```

<table>
  <thead>
    <tr>
      <th>Loss Function Name</th>
      <th>Reprojection Error Image</th>
      <th>Point Cloud Image</th>
      <th>Total Points</th>
      <th>Time Taken</th>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="5" style="text-align:center; font-weight:bold; background:#f0f0f0;"><b>TRF-Based Bundle Adjustment</b></td></tr>
    <tr>
      <td>Linear</td>
      <td><img src="https://github.com/user-attachments/assets/6c5fb084-e4d7-4d44-b36a-ff248f82931e" width="200"/></td>
      <td><img src="https://github.com/user-attachments/assets/49ab6c73-9e1d-4860-bebe-6ec6c3c97cea" width="200"/></td>
      <td>16619</td>
      <td>00:04:46</td>
    </tr>
    <tr>
      <td>Huber</td>
      <td><img src="https://github.com/user-attachments/assets/dddf3a65-8893-41ac-857c-0e0e018d39fb" width="200"/></td>
      <td><img src="https://github.com/user-attachments/assets/8ab5bf01-174d-4b26-9acc-892a019c990c" width="200"/></td>
           <td>15419</td>
<td>00:02:24</td>
    </tr>
    <tr>
      <td>Soft L1</td>
      <td><img src="https://github.com/user-attachments/assets/3087e7f5-df53-4350-82cb-9c1e7ba8504e" width="200"/></td>
      <td><img src="https://github.com/user-attachments/assets/73285648-ff51-4490-9533-06a25d451cfe" width="200"/></td>
      <td>15419</td>
      <td>00:02:20</td>
    </tr>
   <tr>
    <td>Cauchy <br/><small><em>(also used in <a href="https://github.com/hassaanahmed04/sfm/tree/main/Ceres%20Solver/main.cpp">Ceres Solver</a>)</em></small></td>
    <td><img src="https://github.com/user-attachments/assets/f8081184-f364-43ca-a0b0-3196c3e55470" width="200"/></td>
    <td><img src="https://github.com/user-attachments/assets/e311bd09-a019-4066-859f-fc0a11733b04" width="200"/></td>
    <td>17626</td>
    <td>00:03:05</td>
   </tr>
    <tr><td colspan="5" style="text-align:center; font-weight:bold; background:#f0f0f0;"><b>Dogbox</b></td></tr>
<tr>
      <td>Soft L1</td>
      <td><img src="https://github.com/user-attachments/assets/2e846fa3-1631-4166-aa8f-ff7438aae003" width="200"/></td>
      <td><img src="https://github.com/user-attachments/assets/a388a8a5-fd1a-46b6-9d5d-7e8a672e56ae" width="200"/></td>
      <td>17626</td>
      <td>00:02:34</td>
    </tr>
      <tr>
      <td>Huber</td>
      <td><img src="https://github.com/user-attachments/assets/20645b17-c827-4b89-a39e-f299ada798dc" width="200"/></td>
      <td><img src="https://github.com/user-attachments/assets/377fd643-a3f4-45fa-b165-9884c77bdb2c" width="200"/></td>
           <td>17626</td>
<td>00:02:20</td>
    </tr>
    <tr><td colspan="5" style="text-align:center; font-weight:bold; background:#f0f0f0;"><b>Levenberg–Marquardt (LM) algorithm</b>b></td></tr>
<tr>
      <td>Linear</td>
      <td><img src="https://github.com/user-attachments/assets/aa2dae84-9c63-46cd-a181-38adf334925c" width="200"/></td>
      <td><img src="https://github.com/user-attachments/assets/c7631bc3-49db-44c3-ac7a-be5e40188df0" width="200"/></td>
      <td>15419</td>
      <td>00:03:31</td>
    </tr>
  </tbody>
</table>


Bundle adjustment using TRF and LM methods yields reliable results, but performance depends on trade-offs between speed, accuracy, and parameter tuning. TRF with linear or Cauchy loss was fastest and most accurate, while LM was slower but still effective. Proper tuning is key for optimal results.

## Final Output using TRF with Cauchy Loss Function 
![Gustave](https://github.com/user-attachments/assets/01a69878-a54c-4080-8891-473edf8c5e20)

## Ceres Implementation

We also implemented bundle adjustment using the **Ceres Solver**. Although the current reprojection error remains high, we are actively working on improving it.

Our implementation differs slightly: we hosted the Ceres bundle adjustment code locally using the **Crow** web framework to create a server, which we call from our main pipeline. 

You can run the Ceres-based optimization via the provided local [link](https://github.com/hassaanahmed04/sfm/tree/main/Ceres%20Solver).

## Camera Calibration for Custom Dataset

Calibrating our smartphones was crucial for applying **Structure from Motion (SfM)** to our dataset. Guided by a YouTube video titled [*“Structure from Motion: How to Process SfM Datasets,”*](https://www.youtube.com/watch?v=ViUILwG9jlo&t=1s)

We used two devices for data collection and calibration:

- **OPPO RENO 5**: [Calibration Dataset (16 images)](https://)  
  - We fixed the following camera settings for consistency (used where applicable):  
    - **ISO**: 100  
    - **Shutter Speed**: 1/20  
    - **Focus**: 0.78  
    - **White Balance**: 5000
- **iPhone 8 Plus**: [Calibration Dataset (37 images)](https://)  

Images of a checkerboard pattern were captured from multiple angles for each device to compute the **intrinsic camera parameters** required for accurate 3D reconstruction.

---

## Creating a Custom Dataset

Generating our dataset was the most challenging part of the project, involving both **indoor (controlled)** and **outdoor (natural)** environments.

![Duckg](https://github.com/user-attachments/assets/28a2f62e-059b-4c11-b513-7bf50fa01395)

### Preprocessing Step

In both indoor and outdoor datasets, we can **remove the background** from the images as an optional **preprocessing step** to improve reconstruction accuracy. While not mandatory, it significantly reduced noise and improved point cloud clarity.

### Closed Environment (Indoor)

- Phone fixed in position  
- Stabilized lighting conditions  
- Object placed on a rotating plate with marked intervals
- Images captured at each interval while the object rotated and the camera remained stationary

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/0d409065-7341-4596-9005-a83401702a41" width="300"/></td>
    <td><img src="https://github.com/user-attachments/assets/b0739279-15f4-4584-8820-aba159665287" width="300"/></td>
    <td><img src="https://github.com/user-attachments/assets/5065d514-d455-4a83-9580-c295304230aa" width="300"/></td>
  </tr>
  <tr>
    <td align="center"><strong>Duck(66 Images)</strong></td>
    <td align="center"><strong>Box(70 Images)</strong></td>
    <td align="center"><strong>Tin(18 Images)</strong></td>
  </tr>
</table>

### Open Environment (Outdoor)

- Object static  
- Phone moved around the object to capture images from all angles
<table align="center">
 <tr>
    <td><img src="https://github.com/user-attachments/assets/89f96df4-ab2b-4f0d-a52a-7e4fdbbcb5eb" width="300"/></td>
    <td><img src="https://github.com/user-attachments/assets/e935ca8e-aa67-432b-9100-4d14c305c95d" width="300"/></td>
  </tr>
  <tr>
    <td align="center"><strong>Fountain(62 Images)</strong></td>
    <td align="center"><strong>Condorcet(43 Images)</strong></td>
  </tr>
  </table>

In all cases, we maintained the **same camera settings** as during calibration to ensure consistent and reliable reconstruction.
