import numpy as np
from keras import ImageDataGenerator

def get_rand_bbox(width, height, l):
    r_x = np.random.randint(width)
    r_y = np.random.randint(height)
    r_l = np.sqrt(1 - l)
    r_w = np.int(width * r_l)
    r_h = np.int(height * r_l)
    return r_x, r_y, r_l, r_w, r_h

def one_hot(y):
    root_binarizer = sklearn.preprocessing.LabelBinarizer()
    root_binarizer.fit(range(168))
    root = root_binarizer.transform(y[0])

    vowel_binarizer = sklearn.preprocessing.LabelBinarizer()
    vowel_binarizer.fit(range(11))
    vowel = vowel_binarizer.transform(y[1])

    cons_binarizer = sklearn.preprocessing.LabelBinarizer()
    cons_binarizer.fit(range(7))
    cons = cons_binarizer.transform(y[2])
    
    return [root, vowel, cons]

# custom image data generator
class MyDataGenerator(ImageDataGenerator):
        
        def __init__(self, cutmix_alpha, zoom_range, rotation_range,img = [], labels = []):
            super().__init__(zoom_range = zoom_range, rotation_range = rotation_range)
            self.cutmix_alpha = cutmix_alpha
            self.img = img
            self.labels = labels
            
        def add_external(self,X,Y):
            if self.img == []:
                return X,Y
            index1 = np.random.randint(len(X), size=10)
            index2 = np.random.randint(len(self.img), size=10)

            X[index1] = self.img[index2]
            Y[0][index1] = self.labels[0][index2]
            Y[1][index1] = self.labels[1][index2]
            Y[2][index1] = self.labels[2][index2]
            
            return X,Y
        
        def cutmix(self, X1, y1, X2, y2):
            '''
            X1, X2 : image of shape (nb_samples, 128,128,1) 
            y1,y2 : output of len 3 
            
            '''
            l = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            width = X1.shape[2]
            height = X1.shape[1]
            r_x, r_y, r_l, r_w, r_h = get_rand_bbox(width, height, l)
            
            bx1 = np.clip(r_x - r_w // 2, 0, width)
            by1 = np.clip(r_y - r_h // 2, 0, height)
            bx2 = np.clip(r_x + r_w // 2, 0, width)
            by2 = np.clip(r_y + r_h // 2, 0, height)
            
            X1[:, bx1:bx2, by1:by2, :] = X2[:, bx1:bx2, by1:by2, :]

            X = X1
            
            l = 1- (by2-by1)*(bx2-bx1)/(width*height)
            y = [l * y1[0]+(1-l)* y2[0], l * y1[1]+(1-l)* y2[1], l * y1[2]+(1-l)* y2[2]]
            
            return X, y
                    
        def myflow(self,df, batch_size, shuffle=False):
            
            orig_flow = super().flow_from_dataframe(
                dataframe = df,
                x_col = "filename",
                y_col = ['grapheme_root','vowel_diacritic','consonant_diacritic'],
                batch_size = batch_size,
                color_mode = "grayscale",
                classes = ['grapheme_root','vowel_diacritic','consonant_diacritic'],
                class_mode = "multi_output",
                shuffle= shuffle,
                target_size=(SIZE,SIZE),
                validate_filenames = False)
            
            if self.cutmix_alpha > 0 :
                while True:      
                    (batch_x, batch_y) = next(orig_flow)
                    while True:
                        batch_x2, batch_y2 = next(orig_flow)
                        m1, m2 = batch_x.shape[0], batch_x2.shape[0]
                        if m1 < m2:
                            batch_x2 = batch_x2[:m1]
                            batch_y2 = [batch_y2[0][:m1],batch_y2[1][:m1],batch_y2[2][:m1]]
                            break
                        elif m1 == m2:
                            break

                    batch_y = one_hot(batch_y)
                    batch_y2 = one_hot(batch_y2)

                    batch_x, batch_y = self.cutmix(batch_x, batch_y, batch_x2, batch_y2)
                    batch_x, batch_y = self.add_external(batch_x, batch_y)

                    yield batch_x, batch_y
            else :
                while True:
                    (batch_x, batch_y) = next(orig_flow)
                    batch_y = one_hot(batch_y)
                    yield batch_x, batch_y