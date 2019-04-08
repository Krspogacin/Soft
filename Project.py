from __future__ import print_function
import cv2
import numpy as np
import tensorflow as tf


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_binary = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_binary


def invert(image):
    return 255-image


def dilate(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


def select_roi(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    cordinates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if area > 50 and h < 100 and h > 15 and w > 3:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cordinates.append((x, y, x+w, y+h))
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # for reg in regions_array:
    #     for index, reg2 in enumerate(regions_array):
    #         if reg[1][0] < reg2[1][0] and reg[1][1] < reg2[1][1] and reg[1][0] + reg[1][2] > reg2[1][0] + reg2[1][2] and reg[1][1] + reg[1][3] > reg2[1][1] + reg2[1][3]:
    #             del regions_array[index]
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    cordinates = sorted(cordinates, key=lambda item: item[0])
    sorted_regions = [region[0] for region in regions_array]

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions, cordinates


def lines(mask):
    edges = cv2.Canny(mask, 50, 150)
    edges = dilate(edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 600, 8)

    correctLine = lines[0]
    for line in lines:
        if line[0][0] < correctLine[0][0]:
            correctLine[0][0] = line[0][0]
            correctLine[0][1] = line[0][1]
        if line[0][3] < correctLine[0][3]:
            correctLine[0][2] = line[0][2]
            correctLine[0][3] = line[0][3]
    return correctLine


def neural_network():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0 # scale to range

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    result = model.evaluate(x_test, y_test)
    print(result)
    return model


def line(x1, y1, x2, y2):
    k = (y2-y1)/(x2-x1)
    n = -x1*k + y1
    return k, n


def line_contact(x, y, k, n):
    if np.abs(k*x + n - y) < 1:
        return True
    else:
        return False


def video(path, model):
    # ucitavanje videa
    frame_num = 0
    cap = cv2.VideoCapture(path)
    cap.set(1, frame_num)  # indeksiranje frejmova
    # analiza videa frejm po frejm
    lineGreen = []
    lineBlue = []
    final_sum = 0
    indAdd = 0
    numberAdd = 0
    indSub = 0
    numberSub = 0
    while True:
        frame_num += 1
        ret_val, frame = cap.read()
        # ako frejm nije zahvacen
        if not ret_val:
            break

        cv2.imshow('original', frame)
        if lineGreen == [] or lineBlue == []:
            mask1 = cv2.inRange(frame, np.array([0, 100, 0]), np.array([100, 255, 100]))
            mask2 = cv2.inRange(frame, np.array([150, 0, 0]), np.array([255, 100, 100]))

            mask1 = dilate(erode(mask1)) # zbog kruzica koji su zelene boje
            mask1 = dilate(mask1)
            mask2 = dilate(mask2)
            lineGreen = lines(mask1)
            lineBlue = lines(mask2)

        cv2.line(frame, (lineGreen[0][0], lineGreen[0][1]), (lineGreen[0][2], lineGreen[0][3]), (0, 0, 255))
        cv2.line(frame, (lineBlue[0][0], lineBlue[0][1]), (lineBlue[0][2], lineBlue[0][3]), (0, 0, 255))

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_gray = image_gray(img)
        img_bin = invert(image_bin(img_gray))
        img_bin = erode(img_bin)
        selected_regions, numbers, cordinates = select_roi(frame.copy(), img_bin)
        # print(cordinates)

        nums = []
        for number in numbers:
            number = invert(number)
            number = erode(number)
            number = number.reshape(1, 28, 28)
            result = model.predict(number)
            nums.append(result.argmax())
        # print(nums)

        kGreen, nGreen = line(lineGreen[0][0], lineGreen[0][1], lineGreen[0][2], lineGreen[0][3])
        kBlue, nBlue = line(lineBlue[0][0], lineBlue[0][1], lineBlue[0][2], lineBlue[0][3])

        for index, cordinate in enumerate(cordinates):
            if line_contact(cordinate[2],cordinate[3],kGreen,nGreen) and cordinate[2] > lineGreen[0][0] and cordinate[2] < lineGreen[0][2]:
                if index != indSub and nums[index] != numberSub:
                    final_sum = final_sum - nums[index]
                    indSub = index
                    numberSub = nums[index]
            if line_contact(cordinate[2],cordinate[3],kBlue,nBlue) and cordinate[2] > lineBlue[0][0] and cordinate[2] < lineBlue[0][2]:
                if index != indAdd and nums[index] != numberAdd:
                    final_sum = final_sum + nums[index]
                    indAdd = index
                    numberAdd = nums[index]
        # print('FINAL:', final_sum)

        # dalje se sa frejmom radi kao sa bilo kojom drugom slikom, npr
        cv2.imshow('after transformations', selected_regions)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    return final_sum


model = neural_network()
sum0 = video('video-0.avi', model)
sum1 = video('video-1.avi', model)
sum2 = video('video-2.avi', model)
sum3 = video('video-3.avi', model)
sum4 = video('video-4.avi', model)
sum5 = video('video-5.avi', model)
sum6 = video('video-6.avi', model)
sum7 = video('video-7.avi', model)
sum8 = video('video-8.avi', model)
sum9 = video('video-9.avi', model)

file = open('out.txt', 'w')
file.write('RA 28/2015 Darko Krspogacin\n'
           'file\tsum\n'
            'video-0.avi\t%d\n'
            'video-1.avi\t%d\n'
            'video-2.avi\t%d\n'
            'video-3.avi\t%d\n'
            'video-4.avi\t%d\n'
            'video-5.avi\t%d\n'
            'video-6.avi\t%d\n'
            'video-7.avi\t%d\n'
            'video-8.avi\t%d\n'
            'video-9.avi\t%d\n' % (sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9))
file.close()
