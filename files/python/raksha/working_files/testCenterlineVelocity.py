import pyvista as pv
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import DBSCAN

# --- Load data ---
centerline = pv.read("1d_model.vtp")
cross_sections = pv.read("1d_model_2_vessels_00100.vtp")

points = cross_sections.points
flowrates = cross_sections['Flowrate']  # Use your actual array name here

# --- Cluster cross-section points ---
clustering = DBSCAN(eps=2.0, min_samples=3).fit(points)
labels = clustering.labels_
print("Unique cluster labels (excluding noise):", np.unique(labels[labels != -1]))
print("Total clusters found:", len(np.unique(labels[labels != -1])))
unique_labels = np.unique(labels[labels != -1])  # ignore noise

section_centroids = []
section_avg_flowrates = []

for label in unique_labels:
    section_points = points[labels == label]
    section_values = flowrates[labels == label]

    centroid = section_points.mean(axis=0)
    avg_flowrate = section_values.mean()

    section_centroids.append(centroid)
    section_avg_flowrates.append(avg_flowrate)

section_centroids = np.array(section_centroids)
section_avg_flowrates = np.array(section_avg_flowrates)

# --- Define centerline as a segment (2 points) ---
p0, p1 = centerline.points[0], centerline.points[1]
line_vec = p1 - p0
line_len_sq = np.dot(line_vec, line_vec)

# --- Project centroids onto line segment ---
projected_points = []

for c in section_centroids:
    vec_pc = c - p0
    t = np.dot(vec_pc, line_vec) / line_len_sq
    t = np.clip(t, 0.0, 1.0)  # Clamp to segment
    projection = p0 + t * line_vec
    projected_points.append(projection)

projected_points = np.array(projected_points)

# --- Save output to .vtp ---
output = pv.PolyData(projected_points)
output["AverageFlowrate"] = section_avg_flowrates
output.save("projected_flowrate_on_segment.vtp")

print("✅ Projected flowrate values saved to 'projected_flowrate_on_segment.vtp'")
