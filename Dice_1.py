import cv2
import numpy as np
import math
import os

# --- Helper function to create placeholder PNGs (same as before) ---
def create_placeholder_images(size=200):
    """Creates three simple PNG files for demonstration."""
    filenames = ["/content/Dice_4.png", "/content/Dice_5.png", "/content/Dice_6.png"] 
    if not os.path.exists('/content'): os.makedirs('/content')
    for i, filename in enumerate(filenames):
        if not os.path.exists(filename):
            print(f"Creating placeholder file: {filename}")
            img = np.full((size, size, 3), (220, 220, 220), dtype=np.uint8)
            color = [(50, 50, 200), (50, 160, 50), (200, 50, 50)][i]
            text = str(i + 1)
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3.5, 8)
            cv2.putText(img, text, ((size - w) // 2, (size + h) // 2), cv2.FONT_HERSHEY_SIMPLEX, 3.5, color, 8)
            cv2.imwrite(filename, img)

def create_2d5_cube(face_paths, theta_deg=45, phi_deg=60, size=200,
                    output_canvas_size=400, perspective_strength=0.5, scale_factor=1.0):
    """
    Creates a 2.5D cube with adjustable perspective.

    Args:
        face_paths (list): A list of three paths to the face images [Top, Front, Left].
        theta_deg (float): Rotation angle around the Y-axis (yaw).
        phi_deg (float): Rotation angle around the X-axis (pitch).
        size (int): The side length of the cube.
        output_canvas_size (int): The size of the final output image.
        perspective_strength (float): Controls the amount of perspective (0=orthographic, 1=strong).
        scale_factor (float): Overall scaling of the final cube.
    """
    
    # --- 1. Load Images (same as before) ---
    images = []
    for path in face_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: raise FileNotFoundError(f"Could not load image from {path}.")
        if img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        images.append(cv2.resize(img, (size, size)))
    img_top, img_front, img_left = images

    # --- 2. Define Cube Centered at Origin (same as before) ---
    s = size / 2
    vertices = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]
    ])
    faces = {
        'top':   {'img': img_top,   'verts': [7, 6, 2, 3]},
        'front': {'img': img_front, 'verts': [4, 5, 1, 0]},
        'left':  {'img': img_left,  'verts': [7, 4, 0, 3]}
    }

    # --- 3. 3D Rotation (same as before) ---
    theta = math.radians(theta_deg)
    phi = math.radians(-phi_deg) # Negative for intuitive "upward" pitch
    Ry = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(phi), -math.sin(phi)], [0, math.sin(phi), math.cos(phi)]])
    R = np.dot(Rx, Ry)
    rotated_vertices = np.dot(vertices, R.T)

    # --- 4. NEW: Perspective Projection ---
    # A larger distance simulates a camera further away, resulting in less perspective distortion
    distance = size * 2
    
    projected_2d = np.zeros((rotated_vertices.shape[0], 2))
    for i, v in enumerate(rotated_vertices):
        x, y, z = v
        # The perspective effect is controlled by how much we divide by z
        # We add 'distance' to prevent division by zero and control the focal length
        # The perspective_strength parameter scales the effect of z
        scale = distance / (distance - z * perspective_strength)
        
        projected_2d[i, 0] = x * scale * scale_factor
        projected_2d[i, 1] = y * scale * scale_factor

    # Shift to canvas center
    projected_2d += output_canvas_size / 2
    
    # --- 5. Z-Sorting and Rendering (same as before) ---
    face_data = []
    for name, face in faces.items():
        avg_z = np.mean(rotated_vertices[face['verts'], 2])
        face_data.append({'name': name, 'img': face['img'], 'dst_pts': projected_2d[face['verts']], 'z': avg_z})
    
    sorted_faces = sorted(face_data, key=lambda f: f['z'])
    final_image = np.zeros((output_canvas_size, output_canvas_size, 4), dtype=np.uint8)

    for face in sorted_faces:
        src_pts = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
        dst_pts = face['dst_pts'].astype(np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_face = cv2.warpPerspective(face['img'], M, (output_canvas_size, output_canvas_size))
        
        alpha_channel = warped_face[:, :, 3] / 255.0
        alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

        final_image[:, :, :3] = (warped_face[:, :, :3] * alpha_mask + final_image[:, :, :3] * (1 - alpha_mask))
        final_image[:, :, 3] = np.maximum(final_image[:, :, 3], warped_face[:, :, 3])

    return final_image

if __name__ == '__main__':
    create_placeholder_images(size=200)

    image_paths = ["/content/Dice_4.png", "/content/Dice_5.png", "/content/Dice_6.png"]

    # --- Compare Orthographic vs. Perspective ---

    # 1. Original Orthographic version
    print("Generating Orthographic Cube (no perspective)...")
    orthographic_cube = create_2d5_cube(
        image_paths,
        perspective_strength=0, # perspective=0 means orthographic
        scale_factor=1.0 # Use scale to make it a similar size to the perspective one
    )
    cv2.imwrite("/content/Cube_Orthographic.png", orthographic_cube)

    # 2. New Perspective version
    print("Generating Perspective Cube...")
    perspective_cube = create_2d5_cube(
        image_paths,
        perspective_strength=0.7, # A fairly strong perspective
        scale_factor=1.2 # Scale it up slightly to fill the canvas more
    )
    cv2.imwrite("/content/Cube_Perspective.png", perspective_cube)

    # Display in Colab or a local window
    try:
        from google.colab.patches import cv2_imshow
        print("\n--- Orthographic (no perspective) ---")
        cv2_imshow(orthographic_cube)
        print("\n--- Perspective (faces 'closer together') ---")
        cv2_imshow(perspective_cube)
    except ImportError:
        cv2.imshow('Orthographic Cube', orthographic_cube)
        cv2.imshow('Perspective Cube', perspective_cube)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
