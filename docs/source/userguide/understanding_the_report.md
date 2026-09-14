# understanding the report

By default ptypy creates a report after the reconstruction finished.
It calculates a few important metrics that are generally worth mentioning when reporting or publishing your ptrychographic results, as they tell the informed reader how well the ptychographic scan was performed, how much redundancy in the raw data exists and how much to trust the results you obtained.

## How to switch off the creation of the report:
If you do not care about the report being created for your reconstruction, it is possible to switch it off via the IO settings in the parameter tree.
The following lines of code within your reconsturction script will prevent the report from being calculated.

```python
p.io.report = u.Param()
p.io.report.metrics = None    # default is 'all'
```

This will save a tiny bit of computing effort post reconstruction.

## Where is the report saved
This report is saved within the last written reconstruction (.ptyr) file.
The 'report' can be found on the same level as the 'obj' and 'probe' data fields.
It is calculated seperatly for each storageID (same as the 'obj' and 'probe').

## What is in the report
The report calcuates the following metrics:

- probe size
- average step size
- the linear and area overlap
- the speckle oversampling
- the size of the imaged field of view
- the number of photons detected
- pixel wise maps of photons on the sample, transmitted through the sample and update frequency

In the following the seperate metrics and their impact are explained in more detail.

### probe size
One of the most important metrics for any ptychographic reconstruction is the size of the probing beam on the sample.
With that is meant the lateral extent of the area on the sample that is illuminated / exposed at each scan position.
While it seems like a straight forward thing to calculate, there is no one single way of reporting the probing beam's size in the ptychography community. 
Hence ptypy calculates the probe size in various ways.
Depending on the shape and structure of your probe one way might be more suitable than another.
Keep in mind to report how the probe size was estimated along with the number that you report.

#### Full Width at Half Maximum (FWHM)
Estimating the probe size via the full width half maximum projects the probes intensity along the horizontal and the vertical direction, resulting in a vertical and horizontal profile respectively.
The probe size is then estimate via the width of these one-dimensional profiles measured between those points on the y-axis which are half the maximum intensity.

The report in ptypy contains the horizontal and vertical probe size measured via the FWHM in units of pixels and meters.

Measuring the FWHM assumes a "single contiguous blob of intensity". 
If the intensity within the probe is distrubuted as multiple separate islands of intensity, the reported size via FWHM might not be the most applicable metric.
The estimation is always done along the two principal direction of the pixel grid.
If the probe is for example an elongated ellipse that is slightly tilted, the size estimation via FWHM is not done along the principle axes of the ellipse.

- ToDO: add some figures

#### 90 percent of intensity (90perI)
A more general way of capturing the size of unsually structured probe is the 90 percentile.
Here pixels are sorted by their intensity and pixels are added to the list of pixels that contribute predominantly to the probe until 90% of the total beam intensity is explained via these pixels.
This results in a binary 2D image with pixels that explain 90% of the total intensity using the brightest pixels in the probe and those pixels that together explain 10% or less of the total intensity using the dimmest pixels in the probe.

Measuring the position of the left most and the right most / the highest and the lowest of the bright pixels in the probe can taken as a horizontal / vertical probe size. 
One can further report the number of these bright pixels and, together with the pixel size, calculate an area that these bright pixels cover.

The report in ptypy contains the binary image, the horizontal and veritcal size in pixels and meters, as well as the area covered by bright pixels in pixels and square meters.

- ToDO: add some figures

### average step size