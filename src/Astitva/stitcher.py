import cv2
import numpy as np
import os

class PanaromaStitcher:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.homography_matrices = [] 
    
    def resize_image(self, image):
        h, w = image.shape[:2]
        aspect_ratio = h / w
        new_height = int(800 * aspect_ratio)
        resized = cv2.resize(image, (800, new_height))
        return resized

    def convert_xy(self, x, y):
        global center, f
        xt = f * np.tan((x - center[0]) / f) + center[0]
        yt = (y - center[1]) / np.cos((x - center[0]) / f) + center[1]
        return xt, yt
    
    def cylindrical_projection(self, image):
        global center, f
        h, w = image.shape[:2]

        TransformedImage = np.zeros_like(image)

        f = 800 

        cx, cy = w // 2, h // 2
        center = [cx, cy]  

        for y in range(h):
            for x in range(w):
                theta = (x - cx) / f 
                h_ = (y - cy) / f    

                X = np.sin(theta)
                Y = h_
                Z = np.cos(theta)

                x_cyl = int(f * X / Z + cx)
                y_cyl = int(f * Y / Z + cy)

                if 0 <= x_cyl < w and 0 <= y_cyl < h:
                    TransformedImage[y, x] = image[y_cyl, x_cyl]

        return TransformedImage

    def cylindrical_projection(self, Image):
        global w, h, center, f
        h, w = Image.shape[:2]
        center = [w // 2, h // 2]
        f = 800   
        
        TransformedImage = np.zeros(Image.shape, dtype=np.uint8)
        
        AllCoordinates_of_ti =  np.array([np.array([i, j]) for i in range(w) for j in range(h)])
        ti_x = AllCoordinates_of_ti[:, 0]
        ti_y = AllCoordinates_of_ti[:, 1]
        
        ii_x, ii_y = self.convert_xy(ti_x, ti_y)

        ii_tl_x = ii_x.astype(int)
        ii_tl_y = ii_y.astype(int)

        GoodIndices = (ii_tl_x >= 0) * (ii_tl_x <= (w-2)) * (ii_tl_y >= 0) * (ii_tl_y <= (h-2))

        ti_x = ti_x[GoodIndices]
        ti_y = ti_y[GoodIndices]
        
        ii_x = ii_x[GoodIndices]
        ii_y = ii_y[GoodIndices]

        ii_tl_x = ii_tl_x[GoodIndices]
        ii_tl_y = ii_tl_y[GoodIndices]

        dx = ii_x - ii_tl_x
        dy = ii_y - ii_tl_y

        weight_tl = (1.0 - dx) * (1.0 - dy)
        weight_tr = (dx) * (1.0 - dy)
        weight_bl = (1.0 - dx) * (dy)
        weight_br = (dx) * (dy)
        
        TransformedImage[ti_y, ti_x, :] = ( weight_tl[:, None] * Image[ii_tl_y, ii_tl_x, :] ) + ( weight_tr[:, None] * Image[ii_tl_y, ii_tl_x + 1, :] ) + ( weight_bl[:, None] * Image[ii_tl_y + 1, ii_tl_x, :] ) + ( weight_br[:, None] * Image[ii_tl_y + 1, ii_tl_x + 1, :] )
        min_x = min(ti_x)
        TransformedImage = TransformedImage[:, min_x : -min_x, :]

        return TransformedImage, ti_x-min_x, ti_y
    
    def computeHomography(self, pairs):
        A = []
        number_of_points = len(pairs)
        for i in range(number_of_points):
            x_1, y_1, x_2, y_2 = pairs[i][0], pairs[i][1], pairs[i][2], pairs[i][3]  # Extract x1, y1, x2, y2
            A.append([x_1, y_1, 1, 0, 0, 0, -x_1 * x_2, -y_1 * x_2, -x_2])
            A.append([0, 0, 0, x_1, y_1, 1, -x_1 * y_2, -y_1 * y_2, -y_2])
        
        U, S, V = np.linalg.svd(np.asarray(A))
        H = V[-1, :] / V[-1, -1]
        homography = H.reshape(3, 3)

        return homography


    def dist(self, pair, H):
        
        p1 = np.array([pair[0], pair[1], 1])
        p2 = np.array([pair[2], pair[3], 1])

        p2_estimate = np.dot(H, np.transpose(p1))
        p2_estimate = (1 / p2_estimate[2]) * p2_estimate

        return np.linalg.norm(np.transpose(p2) - p2_estimate)


    def RANSAC(self, point_map, threshold=0.6):
    
        bestInliers = []
        homography = None
        for i in range(1000):
            pairs = [point_map[i] for i in np.random.choice(len(point_map), 4)]

            H = self.computeHomography(pairs)
            inliers = {(c[0], c[1], c[2], c[3])
                    for c in point_map if self.dist(c, H) < 5}

            if len(inliers) > len(bestInliers):
                bestInliers = inliers
                homography = H
                if len(bestInliers) > (len(point_map) * threshold):
                    break

        return homography, bestInliers

    def match(self, i1, i2, direction=None):
        imageSet1 = self.getSIFTFeatures(i1)
        imageSet2 = self.getSIFTFeatures(i2)
        print("Direction:", direction)
        matches = self.flann.knnMatch(
            imageSet2['des'],
            imageSet1['des'],
            k=2
        )
        good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.75 * n.distance:
                good.append((m.trainIdx, m.queryIdx))

        if len(good) > 4:
            pointsCurrent = imageSet2['kp']
            pointsPrevious = imageSet1['kp']
            matchedPointsCurrent = np.float32([pointsCurrent[i].pt for (__, i) in good])
            matchedPointsPrev = np.float32([pointsPrevious[i].pt for (i, __) in good])
            point_map = [[p1[0], p1[1], p2[0], p2[1]] for p1, p2 in zip(matchedPointsCurrent, matchedPointsPrev)]
            
            H, s = self.RANSAC(point_map, threshold=0.6)
            return H
        return None

    def getSIFTFeatures(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        return {'kp': kp, 'des': des}

    def leftshift(self):
        a = self.left_list[0]
        for b in self.left_list[1:]:
            H = self.match(a, b, 'left')
            if H is None:
                raise ValueError("Homography computation failed.")
            self.homography_matrices.append(H)
            xh = np.linalg.inv(H)
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
            ds = ds / ds[-1]
            f1 = np.dot(xh, np.array([0, 0, 1]))
            f1 = f1 / f1[-1]
            xh[0, -1] += abs(f1[0])
            xh[1, -1] += abs(f1[1])
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
            offsety = abs(int(f1[1]))
            offsetx = abs(int(f1[0]))
            dsize = (int(ds[0]) + offsetx, int(ds[1]) + offsety)
            tmp = cv2.warpPerspective(a, xh, dsize)
            tmp[offsety:b.shape[0] + offsety, offsetx:b.shape[1] + offsetx] = b
            a = tmp

        self.leftImage = tmp

    def rightshift(self):
        for each in self.right_list:
            H = self.match(self.leftImage, each, 'right')
            if H is None:
                raise ValueError("Homography computation failed.")
            self.homography_matrices.append(H)
            txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
            txyz = txyz / txyz[-1]
            dsize = (int(txyz[0]) + self.leftImage.shape[1], int(txyz[1]) + self.leftImage.shape[0])
            tmp = cv2.warpPerspective(each, H, dsize)
            tmp = self.mix_and_match(self.leftImage, tmp)
            self.leftImage = tmp

    def mix_and_match(self, leftImage, warpedImage):
        i1y, i1x = leftImage.shape[:2]
        for i in range(0, i1x):
            for j in range(0, i1y):
                try:
                    if np.array_equal(leftImage[j, i], [0, 0, 0]) and np.array_equal(warpedImage[j, i], [0, 0, 0]):
                        warpedImage[j, i] = [0, 0, 0]
                    else:
                        if np.array_equal(warpedImage[j, i], [0, 0, 0]):
                            warpedImage[j, i] = leftImage[j, i]
                        elif not np.array_equal(leftImage[j, i], [0, 0, 0]):
                            bw,gw,rw=warpedImage[j,i]
                            bl,gl,rl=leftImage[j,i]
                            warpedImage[j, i] = [bl,gl,rl]
                except:
                    pass
        return warpedImage
    
    def trim_black_borders(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        x, y, w, h = cv2.boundingRect(coords)
        return image[y:y+h, x:x+w]
    
    def make_panaroma_for_images_in(self, path):
        filenames = sorted([os.path.join(path, fname) for fname in os.listdir(path) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not filenames:
            raise ValueError("No images found in the specified directory.")

        self.images = [cv2.resize(cv2.imread(fname), (600, 500)) for fname in filenames]
        #self.images = [self.resize_image(cv2.imread(fname)) for fname in filenames]
        self.count = len(self.images)
        self.left_list, self.right_list, self.center_im = [], [], None

        self.centerIdx = self.count // 2
        self.center_im = self.images[self.centerIdx]

        for i in range(self.count):
            #transformed_image,_,_ = self.cylindrical_projection(self.images[i])
            #self.images[i] = transformed_image
            if i <= self.centerIdx:
                self.left_list.append(self.images[i])
            else:
                self.right_list.append(self.images[i])

        self.leftshift()
        self.rightshift()

        stitched_result = self.trim_black_borders(self.leftImage)

        return stitched_result, self.homography_matrices

