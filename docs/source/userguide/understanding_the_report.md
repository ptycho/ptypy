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

- ToDO: check that this really works

## Where is the report saved
This report is saved within the last written reconstruction (.ptyr) file.
The 'report' can be found on the same level as the 'obj' and 'probe' data fields.
It is calculated seperatly for each storageID (same as the 'obj' and 'probe').

## What is in the report
The report calcuates the following metrics:

- probe size
- average step size
- the linear and area overlap
- the geometric (speckle) oversampling
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

	../report/metrics/<storageID>/probe_size/FWHM_px
	../report/metrics/<storageID>/probe_size/FWHM_m

Measuring the FWHM assumes a "single contiguous blob of intensity". 
If the intensity within the probe is distrubuted as multiple separate islands of intensity, the reported size via FWHM might not be the most applicable metric.
The estimation is always done along the two principal direction of the pixel grid.
If the probe is for example an elongated ellipse that is slightly tilted, the size estimation via FWHM is not done along the principle axes of the ellipse.

- ToDO: add some figures

#### 90 percent of intensity (90perI)
A more general way of capturing the size of unsually structured probe is the 90 percentile.
Here pixels are sorted by their intensity and pixels are added to the list of pixels that contribute predominantly to the probe until 90% of the total beam intensity is explained via these pixels.
This results in a binary 2D image with pixels that explain 90% of the total intensity using the brightest pixels in the probe and those pixels that together explain 10% or less of the total intensity using the dimmest pixels in the probe.

	content/report/metrics/<storageID>/probe_size/90perI_image

Measuring the position of the left most and the right most / the highest and the lowest of the bright pixels in the probe can taken as a horizontal / vertical probe size. 

	content/report/metrics/<storageID>/probe_size/90perI_width_px
	content/report/metrics/<storageID>/probe_size/90perI_width_m

One can further report the number of these bright pixels and, together with the pixel size, calculate an area that these bright pixels cover.

	content/report/metrics/<storageID>/probe_size/90perI_area_px
	content/report/metrics/<storageID>/probe_size/90perI_area_sqm

The report in ptypy contains the binary image, the horizontal and veritcal size in pixels and meters, as well as the area covered by bright pixels in pixels and square meters.

- ToDO: add some figures

### average step size

Another important metric to report is the (average) step size.
It defines how densily a certain field of view is covered in scan positions.
It should always be smaller than the probe size, as a too large step size will hinder or even prevent the convergance of the ptychographic reconstruction.
Step size and probe size togehter allow for calculating the overlap (see further on in the list of metrics).

While the step size is often already defined in the scan command used to record ptychographic data, ptypy explicitly calculates the average step size from the positions used in, and potentially even refined during, the ptychographic reconstruction.

#### via the three nearest neighbors (from_NN)

To estimate the step size, ptypy finds the three nearest neighbors of each scan point. 
Averaging the all the distances to the three nearest neighbors of each scan point results in a single distance that is reported in meters. 

	content/report/metrics/<storageID>/average_step_size/from_NN_m

In case of different probe size in for example horizontal and vertical direction, it is often advisable to use different step sizes in the horizontal and veritcal direction as well, to keep the relative overlap similar n both directons.
The way pytpty calculates the average step size, does not account for such cases or very inhomegenious sampling patterns like for example often utilized in nearfield ptychography.

- ToDO: add some figures
- ToDO: add references for nearfield


### overlap
The overlap is a measurment of the redundacy in the ptychographic data.
It explains how much the illuminated sample areas of adjacent scan positions overlap, or in other words how much of the previously illuminated sample area is re-illuminated at a neighboring scan point.
The higher the overlap, the more redundancy there is in the ptychographic data and the more likely a successful reconstruction can be obtained.
But high overlap and high redundancy comes at the price of slow scanning speed, as the same areas of the object are illuminated many times from nearby scanning positions.

As usual there is not a single way of reporting the overlap.
Ptypy calculates the linear overlap area overlap.

- ToDO: add references 

#### liner overlap
The linear overlap is calculated as:

```{math}
:label: My label

\mbox{overlap}_{\tiny\mbox{linear}} = 1 - \frac{\mbox{step size}}{\mbox{probe size}}
```

The previously reported average step size is used as step size in this calculation.
As probe size, both the estimates via FWHM as well as the 90 percent of intensity criterion results are being used, resulting in two estimates for the linear overlap.

	content/report/metrics/<storageID>/overlap/linear/from_90perI
	content/report/metrics/<storageID>/overlap/linear/from_FWHM

#### area overlap overlap
The area overlap is calculated as an average over all scan points.
Using the binary map of significantly intense pixels within the probe (see 90 precentile criterion) and the scan positions a map of how often a pixel was (significantly) illumated.
Subtracting one then gives how often a pixel was illuminated again beyond the first time (only if it was ever illuminated in the first place).

The calculating the average of how often pixels were re-illumnated and illuminated in the first place gives an average area overlap.

	content/report/metrics/<storageID>/overlap/area/from_probe_coverage

Calculating the overlap this way takes into account non-regular scanning positions and the overlap through more than just the adjacent scan positions.

### geometric oversampling
The geometric oversampling tells us about how well the detector can sample the speckles within the diffraction patterns. In order to have a chance at a high fidelity reconstruction, the speckles in the diffraction patterns should be sampled by at least two pixels each (Nyquist sampling).

-beam size goes into it
-worse sampling is fine with more steps, incl reference






### size of the imaged field of view
As a scanning technique, ptychography as a theoretically unlimmeted field of view.
Therefore it is important to mention how much sample area was actually imaged within a scan.
Within the report ptypy has two ways of estimating the size of the imaged area.

#### via the probe coverage
The first approach is utlizing the extracted probe shape and structure from the 90 percentile criterion.
Convolving this binary map of significantly illuminated pixels of the probe with the known scan positions results in a map showing how often a pixel in the final object was illuminated by a significantly bright part of the probe at various scan positions.

	content/report/metrics/<storageID>/field_of_view/probe_coverage_image

Counting the pixels that have been illumated at least once by a substanially bright bit of the probe, the imaged field of view can be estimated. Knowing the pixel size, this area can also be expressed in square meters.

	content/report/metrics/<storageID>/field_of_view/probe_coverage_px
	content/report/metrics/<storageID>/field_of_view/probe_coverage_sqm

#### via the convex hull of the scan positions
Another way to estimate the imaged field of view is to measure the area of the convex hull of all scan positions.
In order to do that ptypy first figures out which of the scan points form the convex hull of all scan points.
The report features this list of scan positions.

	content/report/metrics/<storageID>/field_of_view/convex_hull_coords_m

Knowing the polygon defining the convex hull, the area of the imaged field of view can be calculated both in terms of pixels and square meters.

	content/report/metrics/<storageID>/field_of_view/convex_hull_area_px
	content/report/metrics/<storageID>/field_of_view/convex_hull_area_m

If the scan area is unusually shaped, this estimate via the convex hull of scan points can be missleading.

- ToDo: add example image


### number of photons detected
Ptychographic data is eventually limited by the Poisson statistics of the detected signal (photons or electrons).
Hence it is important to know exactly how many photons (or electrons) were ectually detected.

The report thus counts the total number of photons (or electrons) that made it through the sample.

	content/report/metrics/<storageID>/photons/total_detected_photons

This number divided by the total number of scan positions is also reported.

	content/report/metrics/<storageID>/photons/average_detected_photons_per_scan_point

### pixel wise maps
Ptypy also reports various metrics as a function of position on the reconstructed object.

#### fluence on the sample
Instead of just calculating the number of photons per scan point, ptypy reports 2D maps as well.
The first shows where on the reconstructed object how many photons were "deposited" throughout the whole measurments.
It is the convolution of the recovered probe intensity with the known scan positions.

	content/report/metrics/<storageID>/maps/fluence

#### photons transmitted through the sample
Knowing the objects transmission, the number of photons that actually made it through the sample at any given point (and got recorded on the detector) is also reported.

	content/report/metrics/<storageID>/maps/transmission

If an area of the sample was stronger absorbing than other parts of the sample, or if fewer photons were depositited onto that area in the first place, then this will result in a higher noise floor / reduced detection limit for those pixels.

#### view covereage
The final map that is reported counts how often a pixel in the object plane is covered by a 'view', meaning how often it got updated throught a single iteration from different scan positions.

	content/report/metrics/<storageID>/maps/view_coverage

Any pixel with the value zero in this map is never seen by the probe at any of the scan positions. 
Hence there can not be any meaningful information in these pixels.
Any piel with the value of 1 in he view coverage map is only ever covered by the probe at one single scan position.
Without any redundancy in the measurment, the value of these pixels in the reconstruction is also not very trustworthy.
Pixels with a high number get updated from many different scan positions and thus have a higher chance at a high fidelity reconstruction.