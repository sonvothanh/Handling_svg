import cv2
import numpy as np
import math
import os

# --- Helper function to create placeholder PNGs if they don't exist ---
def create_placeholder_images(size=200):
    """Creates three simple PNG files for demonstration."""
    filenames = ["/content/Dice_1.png", "/content/Dice_2.png", "/content/Dice_3.png"]
    
    # Create /content directory if it doesn't exist
    if not os.path.exists('/content'):
        os.makedirs('/content')
        
    for i, filename in enumerate(filenames):
        if not os.path.exists(filename):
            print(f"Creating placeholder file: {filename}")
            # Create a base image (e.g., light gray)
            img = np.full((size, size, 3), (220, 220, 220), dtype=np.uint8)
            # Add a colored number
            color = [(50, 50, 200), (50, 160, 50), (200, 50, 50)][i] # BGR colors
            text = str(i + 1)
            # Calculate text size to center it
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3.5, 8)
            cv2.putText(img, text, ((size - w) // 2, (size + h) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.5, color, 8)
            cv2.imwrite(filename, img)

def create_2d5_cube(face_paths, theta_deg=45, phi_deg=60, size=200, output_canvas_size=400):
    """
    Creates a 2.5D cube from three face images using corrected rotation logic.

    Args:
        face_paths (list): A list of three paths to the face images [Top, Front, Left].
        theta_deg (float): Rotation angle around the Y-axis (yaw).
        phi_deg (float): Rotation angle around the X-axis (pitch).
        size (int): The side length of the cube and source images.
        output_canvas_size (int): The size of the final output image.

    Returns:
        numpy.ndarray: The final image of the 2.5D cube with a transparent background.
    """
    
    # --- 1. Load Images ---
    # We expect images in order: [Top, Front, Left]
    images = []
    for path in face_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) # Load with alpha channel if it exists
        if img is None:
            raise FileNotFoundError(f"Could not load image from {path}.")
        if img.shape[2] == 3: # If no alpha channel, add one
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        images.append(cv2.resize(img, (size, size)))

    img_top, img_front, img_left = images

    # --- 2. Define Cube Centered at Origin ---
    s = size / 2
    vertices = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]
    ])
    
    # Define faces using vertex indices. Order is important for perspective warp!
    # (TL, TR, BR, BL of the flat image)
    faces = {
        'top':   {'img': img_top,   'verts': [7, 6, 2, 3]},  # 2 dots 
        'front': {'img': img_front, 'verts': [3,2,1,0]},  # 1 dot :  ?,?,1,0
        'left':  {'img': img_left,  'verts': [7, 4, 0, 3]}   # 3 dots 
    } 


    # --- 3. 3D Rotation and 2D Projection ---
    theta = math.radians(theta_deg) # Yaw
    phi = math.radians(phi_deg)   # Pitch

    # Rotation matrix around Y-axis (yaw)
    Ry = np.array([
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)]
    ])

    # Rotation matrix around X-axis (pitch)
    # We use a negative angle for phi because in image coordinates, Y is down.
    # This makes a positive phi tilt the cube "back" as expected.
    phi = -phi
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(phi), -math.sin(phi)],
        [0, math.sin(phi), math.cos(phi)]
    ])
    
    # Combine rotations: first yaw, then pitch
    R = np.dot(Rx, Ry)
    
    rotated_vertices = np.dot(vertices, R.T)
    
    # Project onto 2D plane and shift to canvas center
    projected_2d = rotated_vertices[:, :2] + output_canvas_size / 2

    # --- 4. Z-Sorting (to draw back faces first) ---
    face_data = []
    for name, face in faces.items():
        face_verts_3d = rotated_vertices[face['verts']]
        avg_z = np.mean(face_verts_3d[:, 2]) # Use Z-depth for sorting
        face_data.append({
            'name': name,
            'img': face['img'],
            'dst_pts': projected_2d[face['verts']],
            'z': avg_z
        })
    
    sorted_faces = sorted(face_data, key=lambda f: f['z'])

    # --- 5. Warp Images and Combine with Alpha Blending ---
    final_image = np.zeros((output_canvas_size, output_canvas_size, 4), dtype=np.uint8)

    for face in sorted_faces:
        src_pts = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
        dst_pts = face['dst_pts'].astype(np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_face = cv2.warpPerspective(face['img'], M, (output_canvas_size, output_canvas_size))

        # Alpha blending to properly layer the faces
        alpha_channel = warped_face[:, :, 3] / 255.0
        # Create a 3-channel alpha mask for broadcasting
        alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

        # Blend RGB channels
        final_image[:, :, :3] = (warped_face[:, :, :3] * alpha_mask +
                                 final_image[:, :, :3] * (1 - alpha_mask))
        # Combine alpha channels
        final_image[:, :, 3] = np.maximum(final_image[:, :, 3], warped_face[:, :, 3])

    return final_image

if __name__ == '__main__':
    # Create placeholder PNG files if needed
    create_placeholder_images(size=200)

    # Define paths to your photos. The order now matters:
    # 1st image: Top face
    # 2nd image: Front face
    # 3rd image: Left face
    image_paths = [
        "/content/Dice_1.png", # Top
        "/content/Dice_2.png", # Front
        "/content/Dice_3.png"  # Left
    ]

    # --- Parameters ---
    THETA = 315  # Yaw: Turn the cube left/right
    PHI = 240    # Pitch: Tilt the cube back/forward
    IMAGE_SIZE = 200
    CANVAS_SIZE = 400

    # Generate the cube
    cube_image = create_2d5_cube(
        face_paths=image_paths,
        theta_deg=THETA,
        phi_deg=PHI,
        size=IMAGE_SIZE,
        output_canvas_size=CANVAS_SIZE
    )

    # Save the result
    output_filename = "/content/2.5D_Dice_Cube_Corrected.png"
    cv2.imwrite(output_filename, cube_image)
    print(f"Corrected cube image saved to {output_filename}")

    # Display in Colab or a local window
    try:
        from google.colab.patches import cv2_imshow
        cv2_imshow(cube_image)
    except ImportError:
        cv2.imshow('Corrected 2.5D Cube', cube_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
