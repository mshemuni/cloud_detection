# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:52:07 2019

@author: mshem
"""
from dag_cld import env
from dag_cld import ast
from dag_cld import mask
from dag_cld import teacher

logger = env.Logger(blabla=True)
fop = env.File(logger)

pmask = mask.Polar(logger)
ima = ast.Image(logger)

svm = teacher.SVM(logger)
cnn = teacher.CNN(logger)
knn = teacher.KNN(logger)
lr = teacher.LR(logger)
nb = teacher.NB(logger)

coord = ast.Coordinates(logger)
fts = ast.Fits(logger)
asttime = ast.Time(logger)

lat = coord.create("41.2333 degree")
lon = coord.create("39.7833 degree")
ele = 3170
number_of_samples = 500

site = ast.Site(logger, lat, lon, ele)
timeCalc = ast.TimeCalc(logger, site)

mask_coordinates = {"E": ((20, 70), (45, 135)), "S": ((20, 70), (135, 225)),
                    "W": ((20, 70), (225, 315)), "N": ((20, 70), (315, 405)),
                    "ZE": ((70, 90), (45, 135)), "ZS": ((70, 90), (135, 225)),
                    "ZW": ((70, 90), (225, 315)), "ZN": ((70, 90), (315, 405))}

#mask_coordinates = {"S": ((20, 70), (135, 225)), "N": ((20, 70), (315, 405))}
        
def show_data(file):
    data = fts.data(file)
    gray = ima.rgb2gray(ima.array2rgb(data))
    mask = pmask.altaz(gray.shape, mask_coordinates["N"][0],
                       mask_coordinates["N"][1], rev=True)
    masked_data = pmask.apply(gray, mask)
    w_data = ima.find_window(masked_data)
    vec, hog = ima.hog(w_data, show=True, mchannel=False)
    ima.show(hog)
    
def day_night_splitter(directory):
    files = fop.list_in_path("{}/*.gz".format(directory))
    for it, file in enumerate(files):
        _, fn, _ = fop.split_file_name(file)
        loc_time = asttime.str2time(fn, FORMAT="%Y_%m_%d__%H_%M_%S.fits")
        utc = asttime.time_diff(loc_time)
        dp = timeCalc.day_part(utc)
        if dp is not None:
            if dp == 1:
                fop.mv(file, "{}/NIGHT".format(directory))
            elif dp == 0:
                fop.mv(file, "{}/DAY".format(directory))
            else:
                fop.mv(file, "{}/DZ".format(directory))
        logger.log((it + 1)/len(files))

def vec_generator(directory):
        files = fop.list_in_path("{}/*.gz".format(directory))
        for it, file in enumerate(files):
            try:
                data = fts.data(file)
                header = fts.header(file, field="?")
                
                path, fname, _ = fop.split_file_name(file)
                fname = fname.replace(".fits", "").replace(".fit", "")
                
                gray = ima.resize(ima.rgb2gray(ima.array2rgb(data)), "25%")
                
                for direction, coordinates in mask_coordinates.items():
                    
                    header["mask"] = (','.join(
                            [str(elem) for elem in coordinates]),
                          "Altitude range, Azimut range")
                    header["mask_d"] = (direction, "Mask Description")
                    
                    
                    the_mask = pmask.altaz(gray.shape, coordinates[0],
                                           coordinates[1], rev=True)
                    masked_data = pmask.apply(gray, the_mask)
                    windowed_masked_data = ima.find_window(masked_data)
        
                    vec, hog_im = ima.hog(windowed_masked_data, mchannel=False)
                    header["IMAGETYP"] = ("IH", "Vector, Hog or Image")
                    fts.write("{}/{}_{}.fits.gz".format(path, fname,
                              direction),
                              ima.list2array([windowed_masked_data, hog_im]),
                              header=header)
                    header["IMAGETYP"] = ("V", "Vector, Hog or Image")
                    fts.write("{}/{}_{}_vec.fits.gz".format(path, fname,
                              direction),
                              vec, header=header)
            except Exception as e:
                logger.log(e)
                
                
def the_svm(path):
    for direction, coordinates in mask_coordinates.items():
        sky_files = fop.list_in_path("{}/clear/*_{}_vec.fits.gz".format(
        path, direction))
        cld_files = fop.list_in_path("{}/cloud/*_{}_vec.fits.gz".format(
        path, direction))
        
        random_sky_files = svm.random_choices(sky_files, number_of_samples)
        random_cld_files = svm.random_choices(cld_files, number_of_samples)
        
        random_sky_vectors = []
        for random_sky_file in random_sky_files:
            vec = fts.data(random_sky_file)
            random_sky_vectors.append(vec)
        
        random_sky_vectors = ima.list2array(random_sky_vectors)
        
        
        random_cld_vectors = []
        for random_cld_file in random_cld_files:
            vec = fts.data(random_cld_file)
            random_cld_vectors.append(vec)
        
        random_cld_vectors = ima.list2array(random_cld_vectors)
        
        
        whole_classes = svm.class_combiner(
        svm.class_adder(random_sky_vectors, 1),
        svm.class_adder(random_cld_vectors, 0))
        
        whole_classes = svm.shuffle(whole_classes)
        
        training, test = svm.tts(whole_classes, test_size=0.20)
        
        X_train, y_train = training[:, :-1], training[:, -1]
        X_test, y_test = test[:, :-1], test[:, -1]
        
        clsf = svm.classifier(X_train, y_train)
        
        y_predict = svm.predict(clsf, X_test)
        
        res = [direction]
        for method, acc in svm.accuracy(y_test, y_predict).items():
            res.append("{:.4f}".format(acc))
            
        yield res
        
def the_knn(path):
    for direction, coordinates in mask_coordinates.items():
        sky_files = fop.list_in_path("{}/clear/*_{}_vec.fits.gz".format(
        path, direction))
        cld_files = fop.list_in_path("{}/cloud/*_{}_vec.fits.gz".format(
        path, direction))
        
        random_sky_files = svm.random_choices(sky_files, number_of_samples)
        random_cld_files = svm.random_choices(cld_files, number_of_samples)
        
        random_sky_vectors = []
        for random_sky_file in random_sky_files:
            vec = fts.data(random_sky_file)
            random_sky_vectors.append(vec)
        
        random_sky_vectors = ima.list2array(random_sky_vectors)
        
        
        random_cld_vectors = []
        for random_cld_file in random_cld_files:
            vec = fts.data(random_cld_file)
            random_cld_vectors.append(vec)
        
        random_cld_vectors = ima.list2array(random_cld_vectors)
        
        
        whole_classes = svm.class_combiner(
        svm.class_adder(random_sky_vectors, 1),
        svm.class_adder(random_cld_vectors, 0))
        
        whole_classes = svm.shuffle(whole_classes)
        
        training, test = svm.tts(whole_classes, test_size=0.20)
        
        X_train, y_train = training[:, :-1], training[:, -1]
        X_test, y_test = test[:, :-1], test[:, -1]
        clsf = knn.classifier(X_train, y_train, n_neighbors=3)
        y_predict = svm.predict(clsf, X_test)
        
        res = [direction]
        for method, acc in svm.accuracy(y_test, y_predict).items():
            res.append("{:.4f}".format(acc))
            
        yield res

def the_cnn(path):
    for direction, coordinates in mask_coordinates.items():
        sky_files = fop.list_in_path("{}/clear/*_{}.fits.gz".format(
                path, direction))
        cld_files = fop.list_in_path("{}/cloud/*_{}.fits.gz".format(
                path, direction))
        
        random_sky_files = svm.random_choices(sky_files, number_of_samples)
        random_cld_files = svm.random_choices(cld_files, number_of_samples)
        
        whole_data = []
        
        for random_sky_file in random_sky_files:
            the_data = fts.data(random_sky_file)[0]
            whole_data.append([ima.normalize(the_data), 1])
            
        for random_cld_file in random_cld_files:
            the_data = fts.data(random_cld_file)[0]
            whole_data.append([ima.normalize(the_data), 0])
            
        training, test = cnn.tts(whole_data)
        
        X_train = []
        y_train = []
        for data, clss in training:
            X_train.append(data)
            y_train.append(clss)
        
        X_test = []
        y_test = []
        for data, clss in test:
            X_test.append(data)
            y_test.append(clss)
            
        w, h = X_train[0].shape
        
        X_train = ima.list2array(X_train).reshape(-1, w, h, 1)
        y_train = ima.list2array(y_train)
        
        X_test = ima.list2array(X_test).reshape(-1, w, h, 1)
        y_test = ima.list2array(y_test)
        
        clsf = cnn.classifier(X_train, y_train, X_test, y_test, epochs=50, plot=False)
        
        y_predict = cnn.predict(clsf, X_test)
        y_predict = y_predict.astype(int)[:, 0]
        
        res = [direction]
        for method, acc in cnn.accuracy(y_test, y_predict).items():
            res.append("{:.4f}".format(acc))
            
        yield res

def the_lr(path):
    for direction, coordinates in mask_coordinates.items():
        sky_files = fop.list_in_path("{}/clear/*_{}_vec.fits.gz".format(
        path, direction))
        cld_files = fop.list_in_path("{}/cloud/*_{}_vec.fits.gz".format(
        path, direction))
        
        random_sky_files = lr.random_choices(sky_files, number_of_samples)
        random_cld_files = lr.random_choices(cld_files, number_of_samples)
        
        random_sky_vectors = []
        for random_sky_file in random_sky_files:
            vec = fts.data(random_sky_file)
            random_sky_vectors.append(vec)
        
        random_sky_vectors = ima.list2array(random_sky_vectors)
        
        
        random_cld_vectors = []
        for random_cld_file in random_cld_files:
            vec = fts.data(random_cld_file)
            random_cld_vectors.append(vec)
        
        random_cld_vectors = ima.list2array(random_cld_vectors)
        
        
        whole_classes = lr.class_combiner(
        lr.class_adder(random_sky_vectors, 1),
        lr.class_adder(random_cld_vectors, 0))
        
        whole_classes = lr.shuffle(whole_classes)
        
        training, test = lr.tts(whole_classes, test_size=0.20)
        
        X_train, y_train = training[:, :-1], training[:, -1]
        X_test, y_test = test[:, :-1], test[:, -1]
        
        clsf = lr.classifier(X_train, y_train)
        
        y_predict = lr.predict(clsf, X_test)
        
        res = [direction]
        for method, acc in lr.accuracy(y_test, y_predict).items():
            res.append("{:.4f}".format(acc))
            
        yield res
        
def the_nb(path, tp="GAUSSIAN"):
    for direction, coordinates in mask_coordinates.items():
        sky_files = fop.list_in_path("{}/clear/*_{}_vec.fits.gz".format(
        path, direction))
        cld_files = fop.list_in_path("{}/cloud/*_{}_vec.fits.gz".format(
        path, direction))
        
        random_sky_files = nb.random_choices(sky_files, number_of_samples)
        random_cld_files = nb.random_choices(cld_files, number_of_samples)
        
        random_sky_vectors = []
        for random_sky_file in random_sky_files:
            vec = fts.data(random_sky_file)
            random_sky_vectors.append(vec)
        
        random_sky_vectors = ima.list2array(random_sky_vectors)
        
        
        random_cld_vectors = []
        for random_cld_file in random_cld_files:
            vec = fts.data(random_cld_file)
            random_cld_vectors.append(vec)
        
        random_cld_vectors = ima.list2array(random_cld_vectors)
        
        
        whole_classes = nb.class_combiner(
        nb.class_adder(random_sky_vectors, 1),
        nb.class_adder(random_cld_vectors, 0))
        
        whole_classes = nb.shuffle(whole_classes)
        
        training, test = nb.tts(whole_classes, test_size=0.20)
        
        X_train, y_train = training[:, :-1], training[:, -1]
        X_test, y_test = test[:, :-1], test[:, -1]
        
        clsf = nb.classifier(X_train, y_train, tp=tp)
        
        y_predict = nb.predict(clsf, X_test)
        
        res = [direction]
        for method, acc in nb.accuracy(y_test, y_predict).items():
            res.append("{:.4f}".format(acc))
            
        yield res
        
def the_lr_save(path):
    for direction, coordinates in mask_coordinates.items():
        sky_files = fop.list_in_path("{}/clear/*_{}_vec.fits.gz".format(
        path, direction))
        cld_files = fop.list_in_path("{}/cloud/*_{}_vec.fits.gz".format(
        path, direction))
        
        random_sky_files = lr.random_choices(sky_files, number_of_samples)
        random_cld_files = lr.random_choices(cld_files, number_of_samples)
        
        random_sky_vectors = []
        for random_sky_file in random_sky_files:
            vec = fts.data(random_sky_file)
            random_sky_vectors.append(vec)
        
        random_sky_vectors = ima.list2array(random_sky_vectors)
        
        
        random_cld_vectors = []
        for random_cld_file in random_cld_files:
            vec = fts.data(random_cld_file)
            random_cld_vectors.append(vec)
        
        random_cld_vectors = ima.list2array(random_cld_vectors)
        
        
        whole_classes = lr.class_combiner(
        lr.class_adder(random_sky_vectors, 1),
        lr.class_adder(random_cld_vectors, 0))
        
        whole_classes = lr.shuffle(whole_classes)
        
        training, test = lr.tts(whole_classes, test_size=0.20)
        
        X_train, y_train = training[:, :-1], training[:, -1]
        X_test, y_test = test[:, :-1], test[:, -1]
        
        clsf = lr.classifier(X_train, y_train)
        
        clsf.save("./deneme.cls")
    
def all_types(path):
    for epoch in range(50):
        for direction, coordinates in mask_coordinates.items():
            sky_files = fop.list_in_path("{}/clear/*_{}_vec.fits.gz".format(
            path, direction))
            cld_files = fop.list_in_path("{}/cloud/*_{}_vec.fits.gz".format(
            path, direction))
            
            random_sky_files = lr.random_choices(sky_files, number_of_samples)
            random_cld_files = lr.random_choices(cld_files, number_of_samples)
            
            random_sky_vectors = []
            for random_sky_file in random_sky_files:
                vec = fts.data(random_sky_file)
                random_sky_vectors.append(vec)
            
            random_sky_vectors = ima.list2array(random_sky_vectors)
            
            
            random_cld_vectors = []
            for random_cld_file in random_cld_files:
                vec = fts.data(random_cld_file)
                random_cld_vectors.append(vec)
            
            random_cld_vectors = ima.list2array(random_cld_vectors)
            
            
            whole_classes = lr.class_combiner(
            lr.class_adder(random_sky_vectors, 1),
            lr.class_adder(random_cld_vectors, 0))
            
            whole_classes = lr.shuffle(whole_classes)
            
            training, test = lr.tts(whole_classes, test_size=0.20)
            
            X_train, y_train = training[:, :-1], training[:, -1]
            X_test, y_test = test[:, :-1], test[:, -1]
            
            svm_clsf = svm.classifier(X_train, y_train)
            svm_y_predict = svm.predict(svm_clsf, X_test)
            smv_acc = list(svm.accuracy(y_test, svm_y_predict).values())
            
            knn_clsf = knn.classifier(X_train, y_train)
            knn_y_predict = knn.predict(knn_clsf, X_test)
            knn_acc = list(knn.accuracy(y_test, knn_y_predict).values())
            
            lr_clsf = lr.classifier(X_train, y_train)
            lr_y_predict = lr.predict(lr_clsf, X_test)
            lr_acc = list(lr.accuracy(y_test, lr_y_predict).values())
            
            nbga_clsf = nb.classifier(X_train, y_train, tp="GAUSSIAN")
            nbga_y_predict = nb.predict(nbga_clsf, X_test)
            nbga_acc = list(nb.accuracy(y_test, nbga_y_predict).values())
            
            nbbe_clsf = nb.classifier(X_train, y_train, tp="BERNOULLI")
            nbbe_y_predict = nb.predict(nbbe_clsf, X_test)
            nbbe_acc = list(nb.accuracy(y_test, nbbe_y_predict).values())
            
            nbca_clsf = nb.classifier(X_train, y_train, tp="CATEGORICAL")
            nbca_y_predict = nb.predict(nbca_clsf, X_test)
            nbca_acc = list(nb.accuracy(y_test, nbca_y_predict).values())
            
            nbco_clsf = nb.classifier(X_train, y_train, tp="COMPLEMENT")
            nbco_y_predict = nb.predict(nbco_clsf, X_test)
            nbco_acc = list(nb.accuracy(y_test, nbco_y_predict).values())
            
            nbmu_clsf = nb.classifier(X_train, y_train, tp="MULTINOMIAL")
            nbmu_y_predict = nb.predict(nbmu_clsf, X_test)
            nbmu_acc = list(nb.accuracy(y_test, nbmu_y_predict).values())
            
            all_lst = [epoch, direction] + smv_acc + knn_acc + lr_acc
            all_lst += nbga_acc + nbbe_acc + nbca_acc + nbco_acc + nbmu_acc
            all_lst = list(map(str, all_lst))
            yield ", ".join(all_lst)
        
        
def save_svm(path):
    for direction, coordinates in mask_coordinates.items():
        sky_files = fop.list_in_path("{}/clear/*_{}_vec.fits.gz".format(
        path, direction))
        cld_files = fop.list_in_path("{}/cloud/*_{}_vec.fits.gz".format(
        path, direction))
        
        random_sky_files = svm.random_choices(sky_files, number_of_samples)
        random_cld_files = svm.random_choices(cld_files, number_of_samples)
        
        random_sky_vectors = []
        for random_sky_file in random_sky_files:
            vec = fts.data(random_sky_file)
            random_sky_vectors.append(vec)
        
        random_sky_vectors = ima.list2array(random_sky_vectors)
        
        
        random_cld_vectors = []
        for random_cld_file in random_cld_files:
            vec = fts.data(random_cld_file)
            random_cld_vectors.append(vec)
        
        random_cld_vectors = ima.list2array(random_cld_vectors)
        
        
        whole_classes = svm.class_combiner(
        svm.class_adder(random_sky_vectors, 1),
        svm.class_adder(random_cld_vectors, 0))
        
        whole_classes = svm.shuffle(whole_classes)
        
        training, test = svm.tts(whole_classes, test_size=0.10)
        
        X_train, y_train = training[:, :-1], training[:, -1]
        X_test, y_test = test[:, :-1], test[:, -1]
        
        clsf = svm.classifier(X_train, y_train)
        
        svm.save(clsf, "clsf/svm_night_{}.clsf".format(direction))
        
def load_svm(path):
    for direction, coordinates in mask_coordinates.items():
        sky_files = fop.list_in_path("{}/clear/*_{}_vec.fits.gz".format(
        path, direction))
        cld_files = fop.list_in_path("{}/cloud/*_{}_vec.fits.gz".format(
        path, direction))
        
        random_sky_files = svm.random_choices(sky_files, number_of_samples)
        random_cld_files = svm.random_choices(cld_files, number_of_samples)
        
        random_sky_vectors = []
        for random_sky_file in random_sky_files:
            vec = fts.data(random_sky_file)
            random_sky_vectors.append(vec)
        
        random_sky_vectors = ima.list2array(random_sky_vectors)
        
        
        random_cld_vectors = []
        for random_cld_file in random_cld_files:
            vec = fts.data(random_cld_file)
            random_cld_vectors.append(vec)
        
        random_cld_vectors = ima.list2array(random_cld_vectors)
        
        
        whole_classes = svm.class_combiner(
        svm.class_adder(random_sky_vectors, 1),
        svm.class_adder(random_cld_vectors, 0))
        
        whole_classes = svm.shuffle(whole_classes)
        
        training, test = svm.tts(whole_classes, test_size=0.99)
        
        X_train, y_train = training[:, :-1], training[:, -1]
        
        clsf = svm.load("clsf/svm_day_{}.clsf".format(direction))
        
        y_predict = svm.predict(clsf, X_train)
        
        res = [direction]
        for method, acc in svm.accuracy(y_train, y_predict).items():
            res.append("{:.4f}".format(acc))
            
        print(res)

if __name__ == "__main__":
    save_svm("D:/asc/night")