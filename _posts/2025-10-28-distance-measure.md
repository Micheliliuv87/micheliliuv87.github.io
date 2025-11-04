---
layout: distill
title: Distance Measures for Data Science
date: 2025-10-28 01:46:22
description: "Include many distance measures: which come in handy and help me through many of my data science projects"
tags: Notes
categories: Statistics
featured: true
citation: true
authors:
  - name: Micheli Liu
    url: "https://micheliliuv87.github.io/"
    affiliations: 
        name: Emory University ISOM

toc:
  - name: Cosine Type Distance
    subsections:
      - name: Cosine Distance
      - name: Pearson Correlation Distance
      - name: Spearman Correlation Distance (Rank Corr)

  - name: Generalized Distance (Metrics)
    subsections:
      - name: Minkowski Distance
      - name: Euclidean Distance
      - name: Chebyshev Distance (Max abs diff. in Coordinate points)

  - name: Scaled Weighted Distance
    subsections:
      - name: Manhalanobis Distance (Scaled Euclidean)
      - name: Canberra Distance (Weighted Manhattan)
giscus_comments: true

---

### **Notes for Readers:**

1.  **Foundational for Data Science:** These distance measures are the bedrock of many fundamental algorithms. You will use them in:

    - **Clustering** (e.g., K-Means, Hierarchical Clustering)
    - **Classification** (e.g., K-Nearest Neighbors)
    - **Anomaly Detection** (to find outliers)
    - **Recommendation Systems** (to find similar users or items)
    - **Dimensionality Reduction** (e.g., in the core of MDS or t-SNE)

2.  **Ubiquitous in Practice:** The distances covered here include notes and hands-on practices you can try on your own. You will likely encounter them repeatedly in coursework, projects, and real-world applications.

3.  **A Starting Point, Not the Finish Line:** This collection is not exhaustive (it omits, for example, Hamming distance, Jaccard index, and Earth Mover's Distance), but it covers the most distances that are at least used once or multiple times in my school and work projects. So it is a good starting point to begin and learn.

# **Cosine Type Distance**

## **Cosine Distance**
(Almost same direction in high dimentional vectors, and similar angle)

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20200911171455/UntitledDiagram2.png" alt="Cosine Similarity" style="display:block; margin:0 auto; max-width:100%; height:auto; max-height:480px;" />

#### **Definition**

For vectors $\mathbf{s}$ and $\mathbf{t}$ in d-dimensional space, the **cosine similarity** is defined as:

$$\cos(\mathbf{s}, \mathbf{t}) = \frac{\mathbf{s}^\mathsf{T}\mathbf{t}}{\|\mathbf{s}\|_2 \, \|\mathbf{t}\|_2} = \frac{\sum_{j=1}^{d} s_j t_j}{\sqrt{\sum_{j=1}^{d} s_j^2} \cdot \sqrt{\sum_{j=1}^{d} t_j^2}}$$

Then the **cosine distance** is:

$$d(\mathbf{s}, \mathbf{t}) = 1 - \cos(\mathbf{s}, \mathbf{t})$$

- Value Range: Cosine similarity ranges between $[-1, +1]$ **[−1 (perfectly dissimilar) and +1 (perfectly similar)]**. If two vectors have the same direction, similarity $= +1$, distance $= 0$; if they have opposite directions, similarity $= -1$, distance $= 2$(though -1 is less common in many real-world datasets if all values are non-negative).

#### **Why and How It's Used**

1.  Focuses on Orientation, Not Magnitude
    - Cosine measures the angle between vectors, independent of their magnitudes.
    - Even if vectors differ in overall scale, the cosine similarity can remain high if they point in a similar direction.
2.  Common in Text Analysis
    - In high-dimensional sparse vectors like (Bag-of-words) TF–IDF, cosine similarity is high if the directions are similar, regardless of absolute frequencies.
3.  Computationally Simple
    - Only requires the dot product and vector norms, making it relatively efficient to compute.

#### **How It Works (Intuitive Understanding)**

- The more similar the directions of two vectors, the smaller the angle → the greater the cosine similarity → the smaller the cosine distance.
- If vectors are almost orthogonal or diverge, the cosine similarity is small → the cosine distance is large.

#### **Interpretation**

- $vec_b$ is essentially $vec_a * 2$, so they point in the exact same direction.
- Cosine similarity is 1.0 → Cosine distance is 0.

#### **Example:**

```python
import numpy as np

def cosine_distance(s, t):
    s = np.array(s)
    t = np.array(t)

    dot_product = np.dot(s, t)
    norm_s = np.linalg.norm(s) # np.dot: try to write this out in your own practice because dot product is already vectorized.
    norm_t = np.linalg.norm(t)

    # Cosine similarity
    cos_sim = dot_product / (norm_s * norm_t)

    # Cosine distance
    cos_dist = 1 - cos_sim
    return cos_dist

# Example usage:
vec_a = [1, 2, 3]
vec_b = [2, 4, 6]

dist = cosine_distance(vec_a, vec_b)
print("Cosine Distance:", dist)
```

```md
Cosine Distance: 2.0
```

#### **In More Detail**

**Dot Product Part** ($\mathbf{s}^\mathsf{T}\mathbf{t}$ is exactly the **dot product**: $\mathbf{s}^\mathsf{T}\mathbf{t} = \sum_{j=1}^{d} s_j t_j$)

- In English, "$\mathbf{s}^\mathsf{T}\mathbf{t}$" is exactly the dot product of vectors $\mathbf{s}$ and $\mathbf{t}$.
- In this formula, $\mathbf{s}^\mathsf{T}\mathbf{t}$ represents the sum of the coordinate-wise products of vectors $\mathbf{s}$ and $\mathbf{t}$, which is exactly what we call the vector dot product.

**Vector Norm** (This $\|\mathbf{s}\|_2$ and $\|\mathbf{t}\|_2$ are the **Vector Norms**)

- $\|\mathbf{s}\|_2$ is also called the L2 norm (Euclidean norm) of vector $\mathbf{s}$, which can be understood as the vector's length or magnitude.
- Mathematical definition: 

    $$\|\mathbf{s}\|_2 = \sqrt{\sum_{j=1}^{d} s_j^2}$$

- $\|\mathbf{s}\|_2$ represents the length of vector $\mathbf{s}$, calculated by squaring each coordinate, summing these squares, and then taking the square root.

#### **Example:**

```python
import numpy as np

s = np.array([1, 2, 3])
t = np.array([4, 5, 6])

# dot product
dot_value = np.dot(s, t)
print("s • t (dot product):", dot_value)

# norm
norm_s = np.linalg.norm(s)
norm_t = np.linalg.norm(t)
print("||s||_2:", norm_s)
print("||t||_2:", norm_t)

# cosine similarity
cos_sim = dot_value / (norm_s * norm_t)
print("Cosine similarity:", cos_sim)
```

```md
s • t (dot product): 32
||s||\_2: 3.7416573867739413
||t||\_2: 8.774964387392123
Cosine similarity: 0.9746318461970762
```

---

<br>

## **Pearson Correlation Distance**

(The first array appears to be derived by subtracting a constant from the second array)

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20250723175534566635/pearson_correlation_coefficient.webp" alt="Pearson Correlation" style="display:block; margin:0 auto; max-width:100%; height:auto; max-height:480px;" />

#### **Definition**

- The Pearson correlation coefficient measures the **linear** relationship between two vectors.
- Sentence Structure: We first perform mean-centering on vectors $\mathbf{x}$ and $\mathbf{y}$, then compute their cosine similarity.
- English Terms: mean-centering, cosine similarity

#### **Mathematical Expression**

$$
\mathrm{corr}(\mathbf{x}, \mathbf{y})
= \frac{\sum_{j=1}^{d} (x_j - \bar{x})(y_j - \bar{y})}
{\sqrt{\sum_{j=1}^{d} (x_j - \bar{x})^2}\;\sqrt{\sum_{j=1}^{d} (y_j - \bar{y})^2}}
$$

Where

$$\bar{x} = \frac{1}{d}\sum_{j=1}^{d} x_j$$

and,

$$\bar{y} = \frac{1}{d}\sum_{j=1}^{d} y_j$$

#### **Pearson Correlation Distance is defined as**:

$$d_{\text{Pearson}}(\mathbf{x}, \mathbf{y}) = 1 - \mathrm{corr}(\mathbf{x}, \mathbf{y})$$

#### **Why and How It's Used**

1. **Measures Linear Association**
   - Sentence Structure: When you want to know if two vectors have a proportional or inversely proportional linear relationship, you can use the Pearson correlation coefficient.
     - A smaller distance (correlation coefficient closer to 1) indicates higher similarity in linear variation across dimensions.
   - English Terms: linear relationship
2. **Insensitive to Scale and Offset**
   - If one vector is just a linear scaling or shift of another, they can still have a high Pearson correlation coefficient.
3. **Applications in Data Analysis**
   - Feature selection: Find features highly linearly associated with the target variable.
   - Clustering: Group samples with similar linear variation patterns.

#### **How It Works**

- First, subtract the mean from each vector (mean-centering).
- Then perform a calculation similar to cosine similarity, but using the "mean-centered" vectors.
- Finally, use `1 - corr` to get the distance value: higher correlation coefficient ⇒ smaller distance.

#### **Example:**

```python
import numpy as np

def pearson_correlation_distance(x, y):
    # change to float better for calculation
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # calculate mean
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # mean-centering
    x_centered = x - x_mean
    y_centered = y - y_mean

    # dot product
    dot_xy = np.dot(x_centered, y_centered)

    # norm
    norm_x = np.linalg.norm(x_centered)
    norm_y = np.linalg.norm(y_centered)

    # If one vector is all constant(except prime numbers), norm will be 0, thus the correlation distance can be set to 1
    if norm_x == 0 or norm_y == 0:
        return 1.0

    # Pearson correlation
    pearson_corr = dot_xy / (norm_x * norm_y)

    # Pearson distance
    distance = 1 - pearson_corr
    return distance

# Sample data
vec_a = [10, 12, 14, 15]
vec_b = [ 5,  7,  9, 10]

dist = pearson_correlation_distance(vec_a, vec_b)
print("Pearson Correlation Distance:", dist)

# Note: in this case, vec_b appears to be vec_a minus one constant (offsey ~5). If we are expecting they have higher correlation, the distance will be smaller.
```

```md
Pearson Correlation Distance: -2.220446049250313e-16
```

---

<br>

## **Spearman Correlation Distance (Rank Corr)**

<img src="https://statistics.laerd.com/statistical-guides/img/spearman-1-small.png" alt="Spearman Correlation" style="display:block; margin:0 auto; max-width:100%; height:auto; max-height:480px;" />

#### **Definition**

- The Spearman correlation coefficient measures the **rank-based** monotonic relationship between two vectors.
- Sentence Structure: First map the element values of each vector to their respective ranks within the vector, then perform operations similar to Pearson correlation on these "rank vectors."

#### **Mathematical Expression**

- Similar to [Pearson](#pearson-correlation-distance-the-first-array-appears-to-be-derived-by-subtracting-a-constant-from-the-second-array), but instead of mean-centering vectors $\mathbf{x}$ and $\mathbf{y}$, first convert $\mathbf{x}$ to its rank vector, then convert $\mathbf{y}$ to its rank vector, and then calculate the Pearson correlation coefficient.

$$\mathrm{corr}_{\mathrm{Spearman}}(\mathbf{x}, \mathbf{y}) = \mathrm{corr}(\mathrm{rank}(\mathbf{x}),\, \mathrm{rank}(\mathbf{y}))$$

- The corresponding **Spearman Correlation Distance** is then:

$$d_{\text{Spearman}}(\mathbf{x}, \mathbf{y}) = 1 - \mathrm{corr}_{\mathrm{Spearman}}(\mathbf{x}, \mathbf{y})$$

#### **Why and How It's Used**

1. **Focuses on Ranking Rather Than Specific Values**
   - Sentence Structure: As long as the order of values in two vectors is consistent (or nearly consistent), even if their value distributions aren't strictly linear, they can still achieve a high Spearman correlation.
   - English Terms: rank, monotonic
2. **More Robust to Outliers**
   - Because it only considers relative size (which is larger/smaller) and not the absolute magnitude of differences, extreme points don't affect it as much as they do in Pearson correlation.
3. **Used in Non-linear Monotonic Relationship Scenarios**
   - For example, when one vector increases as the other increases (possibly in a curved relationship), Spearman can still capture this monotonic trend.

#### **How It Works**

- Rank the elements of $\mathbf{x}$ and $\mathbf{y}$ from smallest to largest.
- After obtaining the rank vectors, compute the Pearson correlation, then take 1 - correlation to get the distance.

#### **Example:**

```python
import numpy as np

def spearman_correlation_distance(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # make ranks for x,y  (rank)
    x_rank = rank_vector(x)
    y_rank = rank_vector(y)

    # calculate pearson correlation but with x_rank, y_rank
    return pearson_correlation_distance(x_rank, y_rank)

def rank_vector(arr):
    # order ascending by rank and give min rank=1，2nd min rank=2，and so on
    sorted_idx = np.argsort(arr)
    ranks = np.zeros_like(arr)
    for i, idx in enumerate(sorted_idx, start=1):
        ranks[idx] = i
    return ranks

def pearson_correlation_distance(x, y):
    # use the same Pearson equation
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean

    dot_xy = np.dot(x_centered, y_centered)
    norm_x = np.linalg.norm(x_centered)
    norm_y = np.linalg.norm(y_centered)
    if norm_x == 0 or norm_y == 0:
        return 1.0
    corr = dot_xy / (norm_x * norm_y)
    distance = 1 - corr
    return distance

# sample data
vec_a = [10, 100, 20]
vec_b = [5, 90, 10]

dist_spearman = spearman_correlation_distance(vec_a, vec_b)
print("Spearman Correlation Distance:", dist_spearman)

#	in vec_a, order is (10, 20, 100) When they rank，you get rank = (1, 2, 3).
#	in vec_b, order is (5, 10, 90) when they rank, you get rank = (1, 2, 3).
#	Therefore their rank is basically the same, so Spearman distance will be low
```

```md
Spearman Correlation Distance: 2.220446049250313e-16
```

---

<br>

<br>

# **Generalized Distance (Metrics)**

## **Minkowski Distance**

<img src="https://www.kdnuggets.com/wp-content/uploads/c_distance_metrics_euclidean_manhattan_minkowski_oh_12.jpg" alt="Minkowski Distance" style="display:block; margin:0 auto; max-width:100%; height:auto; max-height:480px;" />

#### **Definition**

- For two points $\mathbf{x} = (x_1, x_2, \ldots, x_n)$ and $\mathbf{y} = (y_1, y_2, \ldots, y_n)$ in n-dimensional space, the Minkowski distance is defined as:

$$d(\mathbf{x}, \mathbf{y}) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}}$$

#### **Where $p$ is a parameter that determines the type of distance:**

- When $p = 1$, it yields the Manhattan Distance.
- When $p = 2$, it yields the Euclidean Distance.
- As long as $p \ge 1$, it is a valid Minkowski distance.

#### **Why and How It's Used**

1. _Generalization_: Minkowski distance is a generalized form of many other distance metrics (like Euclidean, Manhattan).
2. _Flexibility_: You can choose different $p$ values based on requirements:
   - $p = 1$: Scenarios that emphasize absolute differences (like city block distance).
   - $p = 2$: The most commonly used straight-line distance.
3. _Machine Learning_: In algorithms like k-Nearest Neighbors (k-NN), you can try different $p$ values to see which distance metric performs better.

#### **How It Works (Intuitive Understanding)**

- First calculate the absolute difference for each coordinate, raise it to the power of $p$, sum these values, and finally take the **p-th root**.
- When $p$ increases, larger differences have a greater impact on the result.

#### **Example:**

```python
import numpy as np

def minkowski_distance(x, y, p=2):
    # x and y are lists or NumPy arrays
    x, y = np.array(x), np.array(y)
    return np.sum(np.abs(x - y) ** p) ** (1 / p)

# Example usage:
point_a = [3, 4, 5]
point_b = [1, 1, 1]

dist_p1 = minkowski_distance(point_a, point_b, p=1)  # Manhattan Distance
dist_p2 = minkowski_distance(point_a, point_b, p=2)  # Euclidean Distance
dist_p3 = minkowski_distance(point_a, point_b, p=3)

print("Minkowski Distance with p=1 (Manhattan):", dist_p1)
print("Minkowski Distance with p=2 (Euclidean):", dist_p2)
print("Minkowski Distance with p=3:", dist_p3)
```

```md
Minkowski Distance with p=1 (Manhattan): 9.0
Minkowski Distance with p=2 (Euclidean): 5.385164807134504
Minkowski Distance with p=3: 4.626065009182741
```

---

<br>

## **Euclidean Distance**

(Point it out, use very often in your data science projects)

<img src="https://rosalind.info/media/Euclidean_distance.png" alt="Euclidean Distance" style="display:block; margin:0 auto; max-width:100%; height:auto; max-height:480px;" />

#### **Definition**

- For two points $\mathbf{x}$ and $\mathbf{y}$ in n-dimensional space, the Euclidean distance (also called L2 distance) is defined as:

$$d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

It is a special case of the Minkowski distance when $p = 2$.

#### Why and How It's Used

1. **Straight-line Distance**: It represents the "straight-line" distance between points in Euclidean space.
2. **Usage in Machine Learning**:
   - In k-Nearest Neighbors (k-NN), it is used to measure similarity or dissimilarity between samples.
   - In clustering (such as k-means), Euclidean distance is often used as a measure of cluster compactness.
3. **Geometric Meaning**: It has a very clear geometric interpretation - the length of the line segment connecting the two points.

#### How It Works (Intuitive Understanding)

- Calculate the difference for each corresponding coordinate, square it, sum up all the squared differences, and then take the square root.
- If the differences are large, the distance will be large as well.

#### **Example:**

```python
import numpy as np

def euclidean_distance(x, y):
    x, y = np.array(x), np.array(y)
    return np.sqrt(np.sum((x - y) ** 2)) #this 1/2 can use sqrt to replace
    #return np.sum((x - y) ** 2) ** (1/2)

point_a = [3, 4, 5]
point_b = [1, 1, 1]

dist_euclidean = euclidean_distance(point_a, point_b)
print("Euclidean Distance:", dist_euclidean)
```

```md
Euclidean Distance: 5.385164807134504
```

---

<br>

## **Chebyshev Distance (Max abs diff. in Coordinate points)**

<img src="https://iq.opengenus.org/content/images/2018/12/chebyshev.png" alt="Chebyshev Distance" style="display:block; margin:0 auto; max-width:100%; height:auto; max-height:480px;" />

#### **Definition**

- For two points $\mathbf{x} = (x_1, x_2, \ldots, x_n)$ and $\mathbf{y} = (y_1, y_2, \ldots, y_n)$ in n-dimensional space, the Chebyshev Distance is defined as:

$$d_{\text{Chebyshev}}(\mathbf{x}, \mathbf{y}) = \max_{1 \le i \le n} \big| x_i - y_i \big|$$

Simply put, it is the maximum of the absolute differences between corresponding coordinates.

#### **Why and How It's Used**

1. **Maximum Difference**:
   - Unlike other distances that "sum" or "average" differences across multiple coordinates, Chebyshev distance only focuses on the largest difference.
   - If one coordinate has a particularly large difference while others have small differences, this single largest difference will determine the overall distance.
2. **Geometric Interpretation**:
   - On a 2D grid (such as a chessboard), Chebyshev distance can be seen as the minimum number of moves a King needs to move from one square to another, since the King can move one step in any direction (including diagonally).
   - In n-dimensional space, this means being able to move 1 unit simultaneously across multiple coordinates, similar to diagonal movement.
3. **Application Scenarios**:
   - Chessboard or grid problems: Calculating the number of moves for a King.
   - Computer games: When diagonal movement is allowed with the same cost as straight movement, Chebyshev distance can measure movement distance.
   - Clustering or anomaly detection: Useful when we want to be highly sensitive to large deviations in any single feature (where a large difference in just one feature makes points distant).

#### **How It Works (Intuitive Understanding)**

- Calculate the absolute difference between corresponding coordinates of the two points: $\lvert x_i - y_i \rvert$.
- Find the maximum value among these differences.
- This maximum value is the Chebyshev distance.

#### **Example:**

```python
import numpy as np

def chebyshev_distance(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.max(np.abs(x - y))

# Example usage:
point_a = [3, 7, 5]
point_b = [1, 1, 10]

dist_chebyshev = chebyshev_distance(point_a, point_b)
print("Chebyshev Distance:", dist_chebyshev)
```

```md
Chebyshev Distance: 6
```

#### **Comparison Between Euclidean/ Manhattan/ Chebyshev Distance**

<img src="https://iq.opengenus.org/content/images/2018/12/distance.jpg" alt="Compare Three Distance" style="display:block; margin:0 auto; max-width:100%; height:auto; max-height:480px;" />

---

<br>

<br>

# **Scaled Weighted Distance**

## **Manhalanobis Distance (Scaled Euclidean)**
(Consider Feature Correlation)

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*KzsugPQU-BTjvDACXbu9qw.jpeg" alt="Manhalanobis Distance1" style="display:block; margin:0 auto; max-width:100%; height:auto; max-height:480px;" />

#### **Definition**

- For two vectors (points) $\mathbf{s}$ and $\mathbf{t}$ in d-dimensional space, the Mahalanobis Distance is defined as:

$$d(\mathbf{s}, \mathbf{t}) = \sqrt{(\mathbf{s} - \mathbf{t})^\mathsf{T} \, \mathbf{C}^{-1} \, (\mathbf{s} - \mathbf{t})}$$

where $\mathbf{C}$ is the covariance matrix of the data, and $\mathbf{C}^{-1}$ is its inverse matrix.

#### **Why and How It's Used**

1. **Accounts for Feature Correlation**:
   - Unlike ordinary Euclidean distance, Mahalanobis distance considers correlations between different features.
   - If two features are highly correlated, the distance in that direction gets "scaled down."
2. **Scale Invariance**:
   - Automatically adjusts for features with different scales based on covariance. If one feature has a particularly large numeric range, it won't dominate the distance measure.
3. **Application Scenarios**:
   - Outlier detection: Points that differ significantly from the data's covariance structure are identified as outliers.
   - Classification/Clustering: When dealing with strongly correlated features, using Mahalanobis distance can more accurately reflect true distance relationships.
4. **Covariates Matching**:
   - This situation requires a reference subject.
   - There's a probability that covariates will skew the distance between subjects.
     - A good solution can be to use the rank of each covariate rather than its value.
     - Or use trimmed mean/median instead of mean.

#### **How It Works (Intuitive Understanding)**

- First calculate the difference between $\mathbf{s}$ and $\mathbf{t}$.
- Use the inverse covariance matrix $\mathbf{C}^{-1}$ to scale this difference, measuring its "degree of difference" in terms of data variance and correlation.
- Finally, take the square root to get the distance.

#### **Example:**

```python
import numpy as np

def mahalanobis_distance(s, t, cov):
    """
    s, t: 1D arrays or lists representing points in d-dimensional space
    cov: covariance matrix (d x d)
    """
    s = np.array(s)
    t = np.array(t)
    diff = s - t
    inv_cov = np.linalg.inv(cov)  # Invert the covariance matrix
    dist = np.sqrt(diff.T @ inv_cov @ diff) # @ is the matrix multiplication operator
    return dist

# Example usage:
point_a = [2, 3] # 2D point can also be np.array([[2, 3],[4,5]]) try yourself
point_b = [5, 7]

# Suppose we know/have an estimated covariance matrix for our 2D data:
cov_matrix = np.array([[4, 1],
                       [1, 2]])  # Just an example sometimes you don't know

dist_mahalanobis = mahalanobis_distance(point_a, point_b, cov_matrix)
print("Mahalanobis Distance:", dist_mahalanobis)
```

```md
Mahalanobis Distance: 2.8784916685156974
```

#### **Example 2: Covariate Matching**

```python
import numpy as np

def m_distnace(xi,xj,cov):
    xi = np.array(xi)
    xj = np.array(xj)
    diff = xi - xj
    inv_cov = np.linalg.inv(cov)
    dist = np.sqrt(diff.T @ inv_cov @ diff)
    return dist

cov = np.array([[335.285714, 4.8095238, 3.78571429],
                [4.80952381, 0.23809524, -0.0238095],
                [3.78571429, -0.0238095, 0.28571429]])

# The Treatment column is not a feature for distance computation because it represents a categorical assignment rather than a characteristic.

# Data from the table (Age, College, Male) # covariance is assumed not fixed params
data = np.array([
    [68, 0, 1],  # Subject 1 (reference)
    [60, 1, 0],  # Subject 2
    [65, 0, 1],  # Subject 3
    [76, 1, 1],  # Subject 4
    [44, 0, 0],  # Subject 5
    [34, 0, 0],  # Subject 6
    [28, 0, 1]   # Subject 7
])

# Define xi (Reference Subject, Subject 1)
xi = data[0]  # Reference subject (Subject 1)

# Define xj (Other Subjects)
xj = data[1:]  # Other subjects (Subjects 2 to 7)

# Compute distances from Subject 1 to all other subjects
distances = [mahalanobis_distance(xi, xj, cov) for xj in data[1:]]

# Print results
for i, d in enumerate(distances, start=2):
    print(f"Mahalanobis Distance d_M(1, {i}) = {d:.2f}")
```

```md
Mahalanobis Distance d_M(1, 2) = 2.88
Mahalanobis Distance d_M(1, 3) = 0.23
Mahalanobis Distance d_M(1, 4) = 2.31
Mahalanobis Distance d_M(1, 5) = 2.00
Mahalanobis Distance d_M(1, 6) = 2.36
Mahalanobis Distance d_M(1, 7) = 3.03
```

---

<br>

## **Canberra Distance (Weighted Manhattan)**

(Use for Text analysis or Gene Expression Data)

<img src="https://media.springernature.com/full/springer-static/image/art%3A10.1186%2Fs44147-024-00535-2/MediaObjects/44147_2024_535_Fig11_HTML.png?as=webp" alt="Canberra Distance" style="display:block; margin:0 auto; max-width:100%; height:auto; max-height:480px;" />

#### **Definition**

- For two d-dimensional vectors $\mathbf{s} = (s_1, s_2, …, s_d)$ and $\mathbf{t} = (t_1, t_2, …, t_d)$, the Canberra distance is defined as:

- Standard equation:

$$d(\mathbf{s}, \mathbf{t}) = \sum_{j=1}^{d} \frac{|s_j - t_j|}{|s_j| + |t_j|}$$

- Sometimes with variation, e.g. normalization:

$$d(\mathbf{s}, \mathbf{t}) = \frac{1}{d} \sum_{j=1}^{d} \frac{|s_j - t_j|}{|s_j| + |t_j|}$$

Sometimes a factor of 2 is seen in the formula (either in front or inside), but the core idea is that it's a weighted version of Manhattan distance.

#### **Why and How It's Used**

1. **Weighted by Magnitude**:
   - Each coordinate difference $\lvert s_j - t_j \rvert$ is divided by $\lvert s_j \rvert + \lvert t_j \rvert$.
   - If both $\lvert s_j \rvert$ and $\lvert t_j \rvert$ are large, the difference is scaled down; if one value is small or zero, the difference is amplified.
2. **Very Sensitive to Small Values**:
   - If a coordinate is near 0, then $\lvert s_j \rvert + \lvert t_j \rvert$ is very small, which causes that term's distance value to become large.
   - This makes Canberra distance very sensitive to changes in features with small or zero values.
3. **Application Scenarios**:
   - Data such as text analysis or gene expression, where zero or near-zero counts are very important.
   - Canberra distance can be used when we are more concerned with relative differences rather than absolute differences.

#### **How It Works (Intuitive Understanding)**

- For each coordinate j, first compute the absolute difference $\lvert s_j - t_j \rvert$.
- Then divide by the sum $\lvert s_j \rvert + \lvert t_j \rvert$. If this sum is small, then that coordinate's contribution to the overall distance becomes larger.
- Sum the results for all dimensions to get the Canberra distance.

Compute $\frac{|1 - 2|}{|1| + |2|} + \frac{|10 - 5|}{|10| + |5|} + \frac{|0 - 3|}{|0| + |3|}$.
Each dimension, due to the different magnitudes of the denominators, contributes differently to the overall distance.

#### **Example:**

```python
import numpy as np

def canberra_distance(s, t):
    s = np.array(s)
    t = np.array(t)
    numerator = np.abs(s - t)
    denominator = np.abs(s) + np.abs(t)

    # To avoid division by zero, we can handle 0 in the denominator carefully:
    # We'll replace 0 with a small epsilon or skip that term if both s_j and t_j are 0.
    # Here, let's just do a safe division approach:
    epsilon = 1e-12
    ratio = numerator / (denominator + epsilon)

    return np.sum(ratio)

# Example usage:
point_a = [1, 10, 0]
point_b = [2, 5, 3]

dist_canberra = canberra_distance(point_a, point_b)
print("Canberra Distance:", dist_canberra)
```

```md
Canberra Distance: 1.6666666666661998
```

<br>

---

## **Final Note:**

I believe it is very important to restate that all distances included in this blog post are the common ones that I used at least onece in either one of my school or work projects. So be aware that are many new distance measures that you may encounter in doing your own projects, don't hesitate to learn.

#### **Many Distances Comparison**

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*UBVod31pjOcv41LJrBC7lg.jpeg" alt="Many Distances Comparison" style="display:block; margin:0 auto; max-width:100%; height:auto; max-height:480px;" />

## **Image Reference**

1. [Spearman Correlation](https://statistics.laerd.com/statistical-guides/img/spearman-1-small.png)
2. [Pearson Correlation](https://media.geeksforgeeks.org/wp-content/uploads/20250723175534566635/pearson_correlation_coefficient.webp)
3. [Cosine Similarity](https://media.geeksforgeeks.org/wp-content/uploads/20200911171455/UntitledDiagram2.png)
4. [Minkowski Distance](https://www.kdnuggets.com/wp-content/uploads/c_distance_metrics_euclidean_manhattan_minkowski_oh_12.jpg)
5. [Euclidean Distance](https://rosalind.info/media/Euclidean_distance.png)
6. [Compare Three Distances](https://iq.opengenus.org/content/images/2018/12/distance.jpg)
7. [Chebyshev Distance](https://iq.opengenus.org/content/images/2018/12/chebyshev.png)
8. [Mahalanobis Distance1](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*KzsugPQU-BTjvDACXbu9qw.jpeg)
9. [Many Distances Comparison](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*UBVod31pjOcv41LJrBC7lg.jpeg)
10. [Canberra Distance](https://media.springernature.com/full/springer-static/image/art%3A10.1186%2Fs44147-024-00535-2/MediaObjects/44147_2024_535_Fig11_HTML.png?as=webp)
