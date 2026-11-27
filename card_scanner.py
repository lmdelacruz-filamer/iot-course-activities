import cv2
import numpy as np

SCENE_FILE   = "lithuania_id.jpg"  # photo of the card lying at an angle
OVERLAY_FILE = "lithuania_id_corners.png"  # scene with the detected corners marked
OUTPUT_FILE  = "lithuania_id_flat.png"  # flattened, top-down result

FLAT_WIDTH  = 480  # width of the flattened card in pixels
FLAT_HEIGHT = 300  # height of the flattened card in pixels


def touches_border(contour, width, height, margin=5):
    """True if any contour point sits on the image frame."""
    points = contour.reshape(-1, 2)
    return bool(((points[:, 0] <= margin) | (points[:, 0] >= width - margin - 1) |
                 (points[:, 1] <= margin) | (points[:, 1] >= height - margin - 1))
                .any())


def segmentation_masks(image):
    """Different ways to separate a card from its background."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    yield edges

    _, bright = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for mask in (bright, 255 - bright):
        yield cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
    _, vivid = cv2.threshold(saturation, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for mask in (vivid, 255 - vivid):
        yield cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def card_quads(mask, width, height):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < 0.03 * width * height:
            break
        if area > 0.6 * width * height or touches_border(contour, width, height):
            continue
        perimeter = cv2.arcLength(contour, True)
        for shrink in (0.02, 0.03, 0.05, 0.08):
            points = cv2.approxPolyDP(contour, shrink * perimeter, True)
            if len(points) != 4:
                continue
            corners = order_corners(points.reshape(4, 2).astype(np.float32))
            top = np.linalg.norm(corners[1] - corners[0])
            side = np.linalg.norm(corners[3] - corners[0])
            if top and side and 1.15 <= top / side <= 2.5:
                yield area, corners
                break


def find_card_corners(image):
    """Return the card's four corners as a float32 array ordered
    top-left, top-right, bottom-right, bottom-left."""
    height, width = image.shape[:2]
    best = None
    for mask in segmentation_masks(image):
        for area, corners in card_quads(mask, width, height):
            if best is None or area > best[0]:
                best = (area, corners)
    if best is None:
        raise RuntimeError("No card found: no card-shaped quadrilateral in the image")
    return best[1]


def order_corners(points):
    """Reorder arbitrary corner positions to TL(Top Left), TR(Top Right), BR(Bot Right), BL(Bot Left)."""
    sums = points.sum(axis=1)
    diffs = points[:, 1] - points[:, 0]
    return np.float32([points[np.argmin(sums)], points[np.argmin(diffs)],
                       points[np.argmax(sums)], points[np.argmax(diffs)]])


def flatten(image, corners):
    """Warp the quad in `corners` into a clean top-down rectangle."""
    destination = np.float32([[0, 0], [FLAT_WIDTH, 0],
                              [FLAT_WIDTH, FLAT_HEIGHT], [0, FLAT_HEIGHT]])
    transform = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, transform, (FLAT_WIDTH, FLAT_HEIGHT))


def main():
    scene = cv2.imread(SCENE_FILE)
    if scene is None:
        raise SystemExit(f"{SCENE_FILE} not found: put a photo of your card "
                         "next to the script and update SCENE_FILE")

    corners = find_card_corners(scene)

    overlay = scene.copy()
    labels = ["TL", "TR", "BR", "BL"]
    for point, label in zip(corners, labels):
        x, y = point.astype(int)
        cv2.circle(overlay, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(overlay, label, (x + 14, y - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    cv2.LINE_AA)
    cv2.imwrite(OVERLAY_FILE, overlay)

    flat = flatten(scene, corners)
    cv2.imwrite(OUTPUT_FILE, flat)

    print("Detected corners (TL, TR, BR, BL):")
    for label, (x, y) in zip(labels, corners):
        print(f"  {label}: ({x:.0f}, {y:.0f})")
    print(f"Overlay saved to {OVERLAY_FILE}")
    print(f"Flattened card saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
