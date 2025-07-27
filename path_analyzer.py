#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import heapq
from typing import List, Tuple, Dict, Set, Optional
import math

class PathAnalyzer:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.original_image = None
        self.processed_image = None
        self.graph = defaultdict(list)
        self.points = {'yellow': [], 'orange': []}
        self.path_pixels = set()
        
    def load_and_preprocess_image(self):
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {self.image_path}")
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        print(f"Image loaded: {self.original_image.shape}")
        
    def detect_colored_points(self):
        hsv = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2HSV)
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        orange_lower = np.array([5, 100, 100])
        orange_upper = np.array([15, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in yellow_contours:
            if cv2.contourArea(contour) > 50:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    self.points['yellow'].append((cx, cy))
        for contour in orange_contours:
            if cv2.contourArea(contour) > 50:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    self.points['orange'].append((cx, cy))
        print(f"Detected {len(self.points['yellow'])} yellow points: {self.points['yellow']}")
        print(f"Detected {len(self.points['orange'])} orange points: {self.points['orange']}")
        
    def extract_path_network(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_not(binary)
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.ximgproc.thinning(binary)
        self.processed_image = binary
        path_coords = np.where(binary == 255)
        self.path_pixels = set(zip(path_coords[1], path_coords[0]))
        print(f"Extracted {len(self.path_pixels)} path pixels")
        
    def build_graph(self):
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for x, y in self.path_pixels:
            neighbors = []
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.path_pixels:
                    distance = math.sqrt(dx*dx + dy*dy)
                    neighbors.append(((nx, ny), distance))
            self.graph[(x, y)] = neighbors
        all_points = self.points['yellow'] + self.points['orange']
        for point in all_points:
            closest_path_pixel = self.find_closest_path_pixel(point)
            if closest_path_pixel:
                distance = math.sqrt((point[0] - closest_path_pixel[0])**2 + (point[1] - closest_path_pixel[1])**2)
                if distance < 20:
                    self.graph[point] = [(closest_path_pixel, distance)]
                    self.graph[closest_path_pixel].append((point, distance))
        print(f"Built graph with {len(self.graph)} nodes")
        
    def find_closest_path_pixel(self, point: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        min_distance = float('inf')
        closest_pixel = None
        for path_pixel in self.path_pixels:
            distance = math.sqrt((point[0] - path_pixel[0])**2 + (point[1] - path_pixel[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_pixel = path_pixel
        return closest_pixel
        
    def dijkstra_shortest_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], float]:
        distances = {node: float('inf') for node in self.graph}
        distances[start] = 0
        previous = {}
        pq = [(0, start)]
        visited = set()
        while pq:
            current_distance, current = heapq.heappop(pq)
            if current in visited:
                continue
            visited.add(current)
            if current == end:
                break
            for neighbor, weight in self.graph[current]:
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        path = []
        current = end
        while current in previous:
            path.append(current)
            current = previous[current]
        path.append(start)
        path.reverse()
        return path, distances[end]

    def find_longest_path_heuristic(self, start: Tuple[int, int], end: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], float]:
        print(f"Finding longest path from {start} to {end}...")
        shortest_path, shortest_dist = self.dijkstra_shortest_path(start, end)
        print(f"Baseline shortest path distance: {shortest_dist:.2f}")
        best_path = shortest_path
        best_distance = shortest_dist
        waypoint_candidates = []
        sample_points = list(self.graph.keys())[::50]
        for waypoint in sample_points[:20]:
            if waypoint == start or waypoint == end:
                continue
            try:
                path1, dist1 = self.dijkstra_shortest_path(start, waypoint)
                path2, dist2 = self.dijkstra_shortest_path(waypoint, end)
                if path1 and path2 and len(path1) > 1 and len(path2) > 1:
                    combined_path = path1 + path2[1:]
                    combined_distance = dist1 + dist2
                    if combined_distance > best_distance:
                        best_path = combined_path
                        best_distance = combined_distance
                        print(f"Found longer path via waypoint {waypoint}: {combined_distance:.2f}")
            except Exception as e:
                continue
        if len(sample_points) > 40:
            print("Trying multi-waypoint paths...")
            for i in range(0, min(10, len(sample_points)-20), 2):
                waypoint1 = sample_points[i]
                waypoint2 = sample_points[i+10]
                if waypoint1 == start or waypoint1 == end or waypoint2 == start or waypoint2 == end:
                    continue
                try:
                    path1, dist1 = self.dijkstra_shortest_path(start, waypoint1)
                    path2, dist2 = self.dijkstra_shortest_path(waypoint1, waypoint2)
                    path3, dist3 = self.dijkstra_shortest_path(waypoint2, end)
                    if all([path1, path2, path3]) and all(len(p) > 1 for p in [path1, path2, path3]):
                        combined_path = path1 + path2[1:] + path3[1:]
                        combined_distance = dist1 + dist2 + dist3
                        if combined_distance > best_distance:
                            best_path = combined_path
                            best_distance = combined_distance
                            print(f"Found longer multi-waypoint path: {combined_distance:.2f}")
                except Exception as e:
                    continue
        print(f"Final longest path distance: {best_distance:.2f} (vs shortest: {shortest_dist:.2f})")
        return best_path, best_distance

    def analyze_paths(self):
        results = {}
        if len(self.points['yellow']) >= 2:
            yellow_start, yellow_end = self.points['yellow'][0], self.points['yellow'][1]
            path, distance = self.dijkstra_shortest_path(yellow_start, yellow_end)
            results['yellow_shortest'] = {
                'path': path,
                'distance': distance,
                'start': yellow_start,
                'end': yellow_end
            }
            print(f"Shortest path between yellow points: {distance:.2f} pixels")
        if len(self.points['orange']) >= 2:
            orange_start, orange_end = self.points['orange'][0], self.points['orange'][1]
            path, distance = self.dijkstra_shortest_path(orange_start, orange_end)
            results['orange_shortest'] = {
                'path': path,
                'distance': distance,
                'start': orange_start,
                'end': orange_end
            }
            print(f"Shortest path between orange points: {distance:.2f} pixels")
        if self.points['yellow'] and self.points['orange']:
            left_yellow = min(self.points['yellow'], key=lambda p: p[0])
            right_orange = max(self.points['orange'], key=lambda p: p[0])
            path, distance = self.find_longest_path_heuristic(left_yellow, right_orange)
            results['yellow_orange_longest'] = {
                'path': path,
                'distance': distance,
                'start': left_yellow,
                'end': right_orange
            }
            print(f"Longest path from left yellow to right orange: {distance:.2f} pixels")
        return results

    def visualize_results(self, results: Dict):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes[0,0].imshow(self.original_image)
        axes[0,0].set_title('Original Image with Detected Points')
        for point in self.points['yellow']:
            axes[0,0].plot(point[0], point[1], 'yo', markersize=10, markeredgecolor='black')
        for point in self.points['orange']:
            axes[0,0].plot(point[0], point[1], 'o', color='orange', markersize=10, markeredgecolor='black')
        axes[0,1].imshow(self.processed_image, cmap='gray')
        axes[0,1].set_title('Extracted Path Network')
        axes[1,0].imshow(self.original_image)
        axes[1,0].set_title('Shortest Paths')
        if 'yellow_shortest' in results:
            path = results['yellow_shortest']['path']
            if path:
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]
                axes[1,0].plot(path_x, path_y, 'y-', linewidth=3, alpha=0.7, label='Yellow shortest')
        if 'orange_shortest' in results:
            path = results['orange_shortest']['path']
            if path:
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]
                axes[1,0].plot(path_x, path_y, 'orange', linewidth=3, alpha=0.7, label='Orange shortest')
        axes[1,0].legend()
        axes[1,1].imshow(self.original_image)
        axes[1,1].set_title('Longest Path (Yellow to Orange)')
        if 'yellow_orange_longest' in results:
            path = results['yellow_orange_longest']['path']
            if path:
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]
                axes[1,1].plot(path_x, path_y, 'red', linewidth=3, alpha=0.7, label='Longest path')
        axes[1,1].legend()
        for ax in axes.flat[::1]:
            for point in self.points['yellow']:
                ax.plot(point[0], point[1], 'yo', markersize=8, markeredgecolor='black')
            for point in self.points['orange']:
                ax.plot(point[0], point[1], 'o', color='orange', markersize=8, markeredgecolor='black')
        plt.tight_layout()
        plt.savefig('path_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig

def main():
    try:
        analyzer = PathAnalyzer('test.jpeg')
        print("=== Path Analysis Started ===")
        print("\n1. Loading and preprocessing image...")
        analyzer.load_and_preprocess_image()
        print("\n2. Detecting colored points...")
        analyzer.detect_colored_points()
        print("\n3. Extracting path network...")
        analyzer.extract_path_network()
        print("\n4. Building graph...")
        analyzer.build_graph()
        print("\n5. Analyzing paths...")
        results = analyzer.analyze_paths()
        print("\n=== RESULTS ===")
        if 'yellow_shortest' in results:
            print(f"\n1. Shortest path between yellow points:")
            print(f"   Distance: {results['yellow_shortest']['distance']:.2f} pixels")
            print(f"   Start: {results['yellow_shortest']['start']}")
            print(f"   End: {results['yellow_shortest']['end']}")
            print(f"   Path length: {len(results['yellow_shortest']['path'])} nodes")
        if 'orange_shortest' in results:
            print(f"\n2. Shortest path between orange points:")
            print(f"   Distance: {results['orange_shortest']['distance']:.2f} pixels")
            print(f"   Start: {results['orange_shortest']['start']}")
            print(f"   End: {results['orange_shortest']['end']}")
            print(f"   Path length: {len(results['orange_shortest']['path'])} nodes")
        if 'yellow_orange_longest' in results:
            print(f"\n3. Longest path from left yellow to right orange:")
            print(f"   Distance: {results['yellow_orange_longest']['distance']:.2f} pixels")
            print(f"   Start: {results['yellow_orange_longest']['start']}")
            print(f"   End: {results['yellow_orange_longest']['end']}")
            print(f"   Path length: {len(results['yellow_orange_longest']['path'])} nodes")
        print("\n6. Generating visualization...")
        analyzer.visualize_results(results)
        print("Results saved as 'path_analysis_results.png'")
        print("\n=== Analysis Complete ===")
        print("\n=== ALGORITHMS USED ===")
        print("\n2. Graph Construction:")
        print("\n3. Path Finding:")
        return results
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
