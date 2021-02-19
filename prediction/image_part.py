def fff(test_image):
    import numpy as np
    from skimage.io import imread
    from skimage.filters import threshold_otsu,laplace
    import matplotlib.pyplot as plt
    from skimage.filters import threshold_otsu
    from skimage.segmentation import clear_border
    from skimage.measure import label
    from skimage.measure import regionprops
    from skimage.color import label2rgb
    from skimage.color import rgb2gray
    from skimage import feature
    import copy
    from skimage import img_as_ubyte
    import cv
    import cv2
    from skimage.segmentation import clear_border
    from skimage.restoration import denoise_tv_chambolle
    x=imread(test_image)
    x = cv2.fastNlMeansDenoisingColored(x,None,10,10,7,21)
    x=denoise_tv_chambolle(x, weight=0.1, multichannel=True)
    z = copy.deepcopy(x)
    yy = copy.deepcopy(x)
    find = []
    f3d = []
    yy = rgb2gray(yy)
    imag = rgb2gray(x) 
    global_thresh = threshold_otsu(imag)
    binary_global = imag > global_thresh
    print(binary_global)
    x = ((feature.canny(yy,sigma=2)))
    plt.imshow(x)
    plt.show()
    y=x.copy()
    clear_border(x)
    label_image = label(x)
    borders = np.logical_xor(x, y)
    label_image[borders] = -1
    image_label_overlay = label2rgb(label_image, image=imag)
    for region in regionprops(label_image):

        if region.area < 10:
           continue
        minr, minc, maxr, maxc = region.bbox
        if 1.5*(maxc - minc) > (maxr - minr) or 6*(maxc - minc) < (maxr - minr):
            continue
        try:
            find.append(copy.deepcopy(binary_global[(minr):maxr,minc:maxc]))
            f3d.append(copy.deepcopy(z[(minr):maxr,minc:maxc,:]))
        except Exception:
            pass
    width = 32
    height = 32
    ddd = []
    eee = []
    img_stack_sm = np.zeros((width, height , 3))
    print(len(find))
    for idx in range(len(find)):
        img = find[idx]
        img_sm = cv2.resize(img_as_ubyte(img), (width, height), interpolation=cv2.INTER_CUBIC)
        img_stack_sm[:,:,0] = copy.deepcopy(cv2.resize(f3d[idx][:,:,0], (width, height), interpolation=cv2.INTER_CUBIC))
        img_stack_sm[:,:,1] = copy.deepcopy(cv2.resize(f3d[idx][:,:,1], (width, height), interpolation=cv2.INTER_CUBIC))
        img_stack_sm[:,:,2] = copy.deepcopy(cv2.resize(f3d[idx][:,:,2], (width, height), interpolation=cv2.INTER_CUBIC))
        img_sm=denoise_tv_chambolle(img_sm, weight=0.1, multichannel=True)
        ddd.append(img_sm)
        eee.append(copy.deepcopy(img_stack_sm))
        plt.imshow(eee[-1])
        plt.show()
    return eee,ddd
#fff('../xxx/20.jpg')