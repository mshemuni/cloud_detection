# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:45:24 2019

@author: mshem
"""
from datetime import datetime
from datetime import timedelta

from matplotlib import pyplot as plt

from numpy import float64 as f64
from numpy import asarray as ar
from numpy import dstack
from numpy import unique
from numpy import cos
from numpy import arccos
from numpy import ones
from numpy import ceil
from numpy import where

from astropy.stats import histogram as hist

from astropy import units

from astropy.io import fits as fts
from astropy.table import Table
from astropy.coordinates import EarthLocation
from astropy.coordinates import SkyCoord
from astropy.coordinates import AltAz
from astropy.coordinates import Angle

from astropy.coordinates import get_sun
from astropy.coordinates import get_moon

from astropy.time import Time as tm

from astroplan import Observer
from astroplan import download_IERS_A

from skimage.color import rgb2gray as r2g
from skimage.feature import hog as the_hog
from skimage.exposure import rescale_intensity as ri

from sep import extract
from sep import Background


class Update:
    """Updates modules"""
    def __init__(self, logger):
        self.logger = logger

    def astroplan(self):
        """Updates astropy"""
        try:
            download_IERS_A()
        except Exception as excpt:
            self.logger.log(excpt)


class Image:
    """Image Class"""
    def __init__(self, logger):
        self.logger = logger

    def find_window(self, array):
        """Returns best fit coordinates of a given masked array"""
        self.logger.log("Finding the best fit for given masked data")
        try:
            result = where(array > 0)
            x_min = result[0].min()
            x_max = result[0].max()
            y_min = result[1].min()
            y_max = result[1].max()
            return array[x_min:x_max, y_min:y_max]

        except Exception as excpt:
            self.logger.log(excpt)

    def blank_image(self, shape, fill=0):
        """Creates a blank array with given shape"""
        self.logger.log("Create blank image")
        try:
            return ones(ceil(shape).astype(int)) * fill
        except Exception as excpt:
            self.logger.log(excpt)

    def unique(self, array):
        """Get uniques from array"""
        self.logger.log("Getting uniques")
        try:
            return unique(array)
        except Exception as excpt:
            self.logger.log(excpt)

    def histogram(self, array, bins=20):
        """Create histogram"""
        try:
            return hist(array, bins=bins)
        except Exception as excpt:
            self.logger.log(excpt)

    def list2array(self, lst):
        """Convert list to array"""
        self.logger.log("Converting list to array")
        try:
            return ar(lst)
        except Exception as excpt:
            self.logger.log(excpt)

    def flatten(self, array):
        """Flatten the array"""
        self.logger.log("Flattening array")
        try:
            return array.flatten()
        except Exception as excpt:
            self.logger.log(excpt)

    def normalize(self, array, lindex=0):
        """Normalize array"""
        self.logger.log("Normalizing array")
        try:
            ret = []
            for layer in range(array.shape[lindex]):
                if lindex == 0:
                    the_array = array[layer]
                elif lindex == 1:
                    the_array = array[:, layer, :]
                elif lindex == 2:
                    the_array = array[:, :, layer]
                else:
                    break
                new_layer = (the_array - the_array.min()) / (
                    the_array.max() - the_array.min())
                ret.append(new_layer)

            ret = ar(ret)
            return ret
        except Exception as excpt:
            self.logger.log(excpt)

    def array2rgb(self, array):
        """Convert a 3D array to RGB array"""
        self.logger.log("Converting 3D array to RGB")
        try:
            if array.ndim == 3:
                if array.shape[0] == 3:
                    return dstack((array[0], array[1], array[2]))
                else:
                    self.logger.log("No enough layers")
            else:
                self.logger.log("A 3D array expected. Got {}D".format(
                    array.ndim))
        except Exception as excpt:
            self.logger.log(excpt)

    def show(self, array, add_points=None):
        """Show the image"""
        self.logger.log("Displaying array")
        try:
            if add_points is not None:
                colors = ["red", "green", "blue", "cyan"]
                for it, point in enumerate(add_points):
                    plt.scatter(point[0], point[1], s=20, c=colors[it])
            plt.imshow(array)
            plt.axis('off')
            plt.show()
        except Exception as excpt:
            self.logger.log(excpt)

    def rgb2gray(self, rgb):
        """Convert RGB array to grayscale array"""
        self.logger.log("Converting RGB to Grayscaled")
        try:
            return r2g(rgb)
        except Exception as excpt:
            self.logger.log(excpt)

    def image2array(self, image):
        """Convert image object to array"""
        self.logger.log("Converting image to array")
        try:
            return ar(image)
        except Exception as excpt:
            self.logger.log(excpt)

    def hog(self, image, show=False, mchannel=True):
        """Histogram of oriented gradients"""
        self.logger.log("Generating HOG")
        try:
            fd, hog_image = the_hog(image, orientations=8,
                                    pixels_per_cell=(16, 16),
                                    cells_per_block=(1, 1), visualize=True,
                                    multichannel=mchannel)

            if show:
                _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4),
                                             sharex=True, sharey=True)

                ax1.axis('off')
                ax1.imshow(self.normalize(image))
                ax1.set_title('Input image')

                # Rescale histogram for better display
                hog_image_rescaled = ri(hog_image, in_range=(0, 10))

                ax2.axis('off')
                ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
                ax2.set_title('Histogram of Oriented Gradients')
                plt.show()

            return(fd, hog_image)
        except Exception as excpt:
            self.logger.log(excpt)

    def projection(self, angle):
        """Cos projection"""
        self.logger.log("Finding projection")
        try:
            return cos(angle)
        except Exception as excpt:
            self.logger.log(excpt)

    def rev_projection(self, DRR):
        """Cos reverse projection"""
        self.logger.log("Finding projection")
        try:
            return arccos(DRR)
        except Exception as excpt:
            self.logger.log(excpt)


class Fits:
    """Fits Class"""
    def __init__(self, logger):
        self.logger = logger

    def write(self, dest, data, header=None, overwrite=True):
        """Writes data to a fits file"""
        self.logger.log("Writing data to file({})".format(dest))
        try:
            fts.writeto(dest, data, header=header, overwrite=overwrite)
        except Exception as excpt:
            self.logger.log(excpt)

    def data(self, file):
        """Get data"""
        try:
            hdu = fts.open(file, "readonly")
            d = hdu[0].data
            hdu.close()
            return d.astype(f64)
        except Exception as excpt:
            self.logger.log(excpt)

    def header(self, file, field="*"):
        """Returns header(s) from file"""
        self.logger.log("Getting Header from {}".format(file))
        ret = []
        try:
            hdu = fts.open(file, mode='readonly')
            header = hdu[0].header
            hdu.close()
            for i in header:
                if not i == "":
                    ret.append([i, header[i]])

            if field == "*":
                return ret
            elif field == "?":
                return header
            else:
                return header[field]
        except Exception as excpt:
            self.logger.log(excpt)

    def background(self, data, isar=False):
        """background of data"""
        self.logger.log("Getting Background of data")
        try:
            bkg = Background(data)
            if isar:
                bkg = ar(bkg)
            return bkg
        except Exception as excpt:
            self.logger.log(excpt)

    def star_finder(self, data, threshold=10, bkgext=True):
        """Find sources on array"""
        try:
            if bkgext:
                bkg = self.background(data)
                data = data - bkg
            coords = extract(data, threshold)
            t = Table(coords)
            t.sort("flux", reverse=True)
            return t
        except Exception as excpt:
            self.logger.log(excpt)


class Time:
    """Time Class"""
    def __init__(self, logger):
        self.logger = logger

    def str2time(self, time, FORMAT='%Y-%m-%dT%H:%M:%S.%f'):
        """Converts string to time object"""
        try:
            datetime_object = datetime.strptime(time, FORMAT)
            return datetime_object
        except Exception as excpt:
            self.logger.log(excpt)

    def time_diff(self, time, time_offset=-3, offset_type="hours"):
        """Time difference calculator"""
        if time is not None and time_offset is not None:
            try:
                if "HOURS".startswith(offset_type.upper()):
                    return time + timedelta(hours=time_offset)
                elif "MINUTES".startswith(offset_type.upper()):
                    return time + timedelta(minutes=time_offset)
                elif "SECONDS".startswith(offset_type.upper()):
                    return time + timedelta(seconds=time_offset)

            except Exception as excpt:
                self.logger.log(excpt)
        else:
            self.logger.log("False Type: One of the values is not correct")

    def jd(self, utc):
        """JD calculator"""
        try:
            t = tm(utc, scale='utc')
            return t.jd
        except Exception as excpt:
            self.logger.log(excpt)

    def jd_r(self, jd):
        """JD to Time calculator"""
        try:
            t = tm(jd, format='jd', scale='tai')
            return t.to_datetime()
        except Exception as excpt:
            self.logger.log(excpt)


class TimeCalc(Time):
    def __init__(self, logger, site):
        super().__init__(logger)
        self._obs_ = site.__observer__()

    def __which_corrector__(self, which):
        if "PREVIOUS".startswith(which.upper()):
            return "previous"
        elif "CLOSEST".startswith(which.upper()):
            return "nearest"
        else:
            return "next"

    def __utc_corrector__(self, utc):
        if utc is None:
            return datetime.now()
        else:
            return utc

    def is_night(self, utc):
        try:
            return self._obs_.is_night(tm(utc))
        except Exception as excpt:
            self.logger.log(excpt)

    def midnight(self, utc, jd=False):
        try:
            if jd:
                return self._obs_.midnight(tm(utc)).jd
            else:
                return self._obs_.midnight(tm(utc)).datetime
        except Exception as excpt:
            self.logger.log(excpt)

    def sun_rise_time(self, utc, which="next", jd=False):
        try:
            which = self.__which_corrector__(which)
            sun_rise = self._obs_.sun_rise_time(tm(utc), which=which)
            if jd:
                return sun_rise.jd
            else:
                return sun_rise.datetime
        except Exception as excpt:
            self.logger.log(excpt)

    def sun_set_time(self, utc, which="next", jd=False):
        try:
            which = self.__which_corrector__(which)
            sun_set = self._obs_.sun_set_time(tm(utc), which=which)
            if jd:
                return sun_set.jd
            else:
                return sun_set.datetime
        except Exception as excpt:
            self.logger.log(excpt)

    def moon_rise_time(self, utc, which="next", jd=False):
        try:
            which = self.__which_corrector__(which)
            moon_rise = self._obs_.moon_rise_time(tm(utc))
            if jd:
                return moon_rise.jd
            else:
                return moon_rise.datetime
        except Exception as excpt:
            self.logger.log(excpt)

    def moon_set_time(self, utc, which="next", jd=False):
        try:
            which = self.__which_corrector__(which)
            moon_set = self._obs_.moon_set_time(tm(utc), which=which)
            if jd:
                return moon_set.jd
            else:
                return moon_set.datetime
        except Exception as excpt:
            self.logger.log(excpt)

    def twilight_morning(self, utc, tp="ASTRONOMICAL", which="next", jd=False):
        try:
            which = self.__which_corrector__(which)
            if "CIVIL".startswith(tp.upper()):
                ret = self._obs_.twilight_morning_civil(
                    tm(utc), which=which)
            elif "NAUTICAL".startswith(tp.upper()):
                ret = self._obs_.twilight_morning_nautical(
                    tm(utc), which=which)
            else:
                ret = self._obs_.twilight_morning_astronomical(
                    tm(utc), which=which)

            if jd:
                return ret.jd
            else:
                return ret.datetime

        except Exception as excpt:
            self.logger.log(excpt)

    def twilight_evening(self, utc, tp="ASTRONOMICAL", which="next", jd=False):
        try:
            which = self.__which_corrector__(which)
            if "CIVIL".startswith(tp.upper()):
                ret = self._obs_.twilight_evening_civil(
                    tm(utc), which=which)
            elif "NAUTICAL".startswith(tp.upper()):
                ret = self._obs_.twilight_evening_nautical(
                    tm(utc), which=which)
            else:
                ret = self._obs_.twilight_evening_astronomical(
                    tm(utc), which=which)

            if jd:
                return ret.jd
            else:
                return ret.datetime
        except Exception as excpt:
            self.logger.log(excpt)


class Coordinates:
    """Coordinate Class"""
    def __init__(self, logger):
        self.logger = logger

    def create(self, angle):
        """Convert String to angle"""
        try:
            return Angle(angle)
        except Exception as excpt:
            self.logger.log(excpt)


class Site:
    """Site Class"""
    def __init__(self, logger, lati, long, alti, name="Obervatory"):
        self.logger = logger
        self._lati_ = lati
        self._long_ = long
        self._alti_ = alti
        self._name_ = name
        self.site = self.create()

    def __observer__(self):
        try:
            return Observer(location=self.site, name=self._name_)
        except Exception as excpt:
            self.logger.log(excpt)

    def create(self):
        """Create site"""
        try:
            s = EarthLocation(lat=self._lati_, lon=self._long_,
                              height=self._alti_ * units.meters)
            return s
        except Exception as excpt:
            self.logger.log(excpt)

    def update(self, lati, long, alti):
        """Update Site"""
        try:
            self.site = EarthLocation(lat=lati, lon=long,
                                      height=alti * units.meters)
        except Exception as excpt:
            self.logger.log(excpt)

    def altaz(self, obj, utc):
        """Return AltAz for a given object and time for this site"""
        try:
            frame_of_sire = AltAz(obstime=utc, location=self.site)
            object_alt_az = obj.transform_to(frame_of_sire)
            return object_alt_az
        except Exception as excpt:
            self.logger.log(excpt)


class Obj:
    """Object Class"""
    def __init__(self, logger, ra, dec):
        self.logger = logger
        self._ra_ = ra
        self._dec_ = dec
        self.obj = self.create()

    def create(self):
        """Create Object"""
        try:
            return SkyCoord(ra=self._ra_, dec=self._dec_)
        except Exception as excpt:
            self.logger.log(excpt)

    def update(self, ra, dec):
        """Update Object"""
        try:
            self.obj = SkyCoord(ra=ra, dec=dec)
        except Exception as excpt:
            self.logger.log(excpt)

    def altaz(self, site, utc):
        """Return AltAz for a site object and time for this object"""
        try:
            frame_of_sire = AltAz(obstime=utc, location=site)
            object_alt_az = self.obj.transform_to(frame_of_sire)
            return object_alt_az
        except Exception as excpt:
            self.logger.log(excpt)


class Sun(Obj):
    def __init__(self, logger, time):
        self.logger = logger
        self._time_ = time
        self.obj = self.create()

    def create(self):
        try:
            return get_sun(self._time_)
        except Exception as excpt:
            self.logger.log(excpt)

    def update(self, time):
        try:
            return get_sun(time)
        except Exception as excpt:
            self.logger.log(excpt)


class Moon(Obj):
    def __init__(self, logger, time):
        self.logger = logger
        self._time_ = time
        self.obj = self.create()

    def create(self):
        try:
            return get_moon(self._time_)
        except Exception as excpt:
            self.logger.log(excpt)

    def update(self, time):
        try:
            return get_moon(time)
        except Exception as excpt:
            self.logger.log(excpt)
