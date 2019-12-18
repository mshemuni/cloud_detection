# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:52:07 2019

@author: mshem
"""

from numpy import logical_not as lnot

from dag_cld import env
from dag_cld import ast
from dag_cld import mask
from dag_cld import teacher

logger = env.Logger(blabla=True)
fil = env.File(logger)

fts = ast.Fits(logger)
ima = ast.Image(logger)

gmask = mask.Geometric(logger)
pmask = mask.Polar(logger)

svm = teacher.SVM(logger)

def generate_hos():
    files = fil.list_in_path("E:/data/partial/*_??.fits")
    
    alt_ranges = [(0, 45), (45, 65), (65, 80), (80, 90)]
    az_ranges = [(0, 22.5), (22.5, 45), (45, 67.5), (67.5, 90),
                 (90, 112.5), (112.5, 135), (135, 157.5), (157.5, 180),
                 (180, 202.5), (202.5, 225), (225, 247.5), (247.5, 270),
                 (270, 292.5), (292.5, 315), (315, 337.5), (337.5, 360)]
    
    for itf, file in enumerate(files):
        data = fts.data(file)
        gimage = ima.rgb2gray(ima.array2rgb(data))
        abs_p = fil.abs_path(file)
        path, file_name, ext = fil.split_file_name(abs_p)
        for it1, alt_range in enumerate(alt_ranges):
            for it2, az_range in enumerate(az_ranges):
                fits_path = "{}/{}/{}/{}{}".format(path, it1, it2, file_name, ext)
                hog_path = "{}/{}/{}/{}_hog{}".format(path, it1, it2, file_name, ext)
                vec_path = "{}/{}/{}/{}.vec".format(path, it1, it2, file_name)
                aiza_mask = pmask.altaz(gimage.shape, alt_range, az_range)
                amasked_data = pmask.apply(gimage, aiza_mask)
                c_image = ima.find_window(amasked_data, aiza_mask)
                hogs = ima.hog(c_image, show=False, mchannel=False)
                fts.write(fits_path, c_image)
                fts.write(hog_path, hogs[1])
                fil.save_numpy(vec_path, hogs[0])
                
        logger.log((itf+1) / len(files))
        
def classifier(file=None):
    alt_ranges = ["0, 45", "45, 65", "65, 80", "80, 90"]
    az_ranges = ["0, 22.5", "22.5, 45", "45, 67.5", "67.5, 90",
                 "90, 112.5", "112.5, 135", "135, 157.5", "157.5, 180",
                 "180, 202.5", "202.5, 225", "225, 247.5", "247.5, 270",
                 "270, 292.5", "292.5, 315", "315, 337.5", "337.5, 360"]
    
    res = []
    data = None
    for alt in range(4):
        for az in range(16):
            open_files_all = fil.list_in_path(
                    "E:/data/day/all_open/{}/{}/*.vec".format(alt, az))
            open_files = svm.random_choices(open_files_all, 70)
            o_data = []
            for o_file in open_files:
                o_data.append(fil.read_array(o_file))
                
            cloudy_files_all = fil.list_in_path(
                    "E:/data/day/all_cloudy/{}/{}/*.vec".format(alt, az))
            
            cloudy_files = svm.random_choices(cloudy_files_all, 70)
            
            c_data = []
            for c_file in cloudy_files:
                c_data.append(fil.read_array(c_file))
                
            o_data = ima.list2array(o_data)
            c_data = ima.list2array(c_data)
            
            
            
            partial_files_all = fil.list_in_path(
                    "E:/data/day/partial/{}/{}/*.vec".format(alt, az))
            
            partial_files = svm.random_choices(partial_files_all, 70)
            
            p_data = []
            for p_file in partial_files:
                p_data.append(fil.read_array(p_file))
                
            o_data = ima.list2array(o_data)
            c_data = ima.list2array(c_data)
            p_data = ima.list2array(p_data)
            
            all_data = svm.class_combiner(svm.class_adder(o_data, 1),
                                          svm.class_adder(c_data, 0),
                                          svm.class_adder(p_data, 2))
            
            X_train, X_test, y_train, y_test = svm.tts(all_data, test_size=0.1)
            
            clf = svm.classifier(X_train, y_train)
            
            if file is None:
                with open("{}__{}.res".format(
                        alt_ranges[alt].replace(", ", "_"),
                        az_ranges[az].replace(", ", "_")), "w") as f:
                
                    for c_file in cloudy_files_all:
                        data = fil.read_array(c_file)
                        print(alt_ranges[alt], az_ranges[az],
                              svm.predict(clf, [data])[0], 0)
                        f.write("{}\t{}\t{}\t{}\n".format(
                                alt_ranges[alt], az_ranges[az],
                                svm.predict(clf, [data])[0], 0))
        
                    for o_file in open_files_all:
                        data = fil.read_array(o_file)
                        print(alt_ranges[alt], az_ranges[az],
                              svm.predict(clf, [data])[0], 1)
                        f.write("{}\t{}\t{}\t{}\n".format(
                                alt_ranges[alt], az_ranges[az],
                                svm.predict(clf, [data])[0], 1))
            else:
                alrange = list(map(float, alt_ranges[alt].split(", ")))
                azrange = list(map(float, az_ranges[az].split(", ")))
                data = fts.data(file)
                gimage = ima.rgb2gray(ima.array2rgb(data))
                aiza_mask = pmask.altaz(gimage.shape, alrange, azrange)
                amasked_data = pmask.apply(gimage, aiza_mask)
                c_image = ima.find_window(amasked_data, aiza_mask)
                hogs = ima.hog(c_image, show=False, mchannel=False)
                co = svm.predict(clf, [hogs[0]])[0]
                res.append([aiza_mask, co])

    res  = ima.list2array(res)
                
    if res is not []:
        print(res.shape)
        print(res)
        w, h = res[0][0].shape
        new_array_r = ima.blank_image((w, h))
        new_array_g = ima.blank_image((w, h))
        new_array_b = ima.blank_image((w, h))
        for d in res:
            if d[1] == 1:
                new_array_g += lnot(d[0]).astype(int)
            elif d[1] == 2:
                new_array_r += lnot(d[0]).astype(int)
            elif d[1] == 0:
                new_array_r += lnot(d[0]).astype(int)
              
        fil.save_numpy("red", new_array_r)
        fil.save_numpy("green", new_array_g)
        fil.save_numpy("blue", new_array_b)
                
        ima.show(ima.array2rgb(ima.list2array([new_array_r,
                                               new_array_g,
                                               new_array_b])))
                
if __name__ == "__main__":
#    classifier("E:/data/2017_08_18__12_42_25.fits")
    generate_hos()