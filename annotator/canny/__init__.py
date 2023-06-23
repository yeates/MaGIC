import kornia

class CannyDetector:
    def __call__(self, img, low_threshold=0.1, high_threshold=0.2):
        return kornia.filters.canny(img, low_threshold, high_threshold)
