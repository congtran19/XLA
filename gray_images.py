import numpy as np

def convert_to_grayscale_manual(img_color: np.ndarray, bgr: bool = True) -> np.ndarray:
    """
    Chuyển ảnh màu sang ảnh xám bằng tay, không dùng OpenCV.
    
    Parameters:
        img_color (np.ndarray): Ảnh màu dạng (H, W, 3)
        bgr (bool): Nếu True, giả định ảnh là BGR (như OpenCV). Nếu False, coi là RGB.
        
    Returns:
        img_gray (np.ndarray): Ảnh xám dạng (H, W)
    """
    if len(img_color.shape) != 3 or img_color.shape[2] != 3:
        raise ValueError("Ảnh đầu vào phải có shape (H, W, 3).")

    if bgr:
        # OpenCV dùng BGR: Blue, Green, Red
        B, G, R = img_color[:,:,0], img_color[:,:,1], img_color[:,:,2]
    else:
        # RGB
        R, G, B = img_color[:,:,0], img_color[:,:,1], img_color[:,:,2]
    
    # Công thức ánh xám chuẩn: 0.299 * R + 0.587 * G + 0.114 * B
    img_gray = 0.299 * R + 0.587 * G + 0.114 * B

    # Chuyển sang kiểu uint8 (0-255)
    img_gray = img_gray.astype(np.uint8)
    
    return img_gray
