# Flir Image Extractor

FLIR® thermal cameras like the FLIR ONE® include both a thermal and a visual light camera.
The latter is used to enhance the thermal image using an edge detector.

The resulting image is saved as a jpg image but both the original visual image and the raw thermal sensor data are embedded in the jpg metadata.

This small Python tool/library allows to extract the original photo and thermal sensor values converted to temperatures.

## Requirements

This tool relies on `exiftool`. It should be available in most Linux distributions (e.g. as `perl-image-exiftool` in Arch Linux or `libimage-exiftool-perl` in Debian and Ubuntu).

It also needs the Python packages *numpy* and *matplotlib* (the latter only if used interactively).

```bash
# sudo apt update
# sudo apt install exiftool
# sudo pip install numpy matplotlib
```

```windows 10
pip install pillow numpy matplotlib
```

```EXIF-TOOL DOWNLOAD Detail as follow..
https://exiftool.org/install.html
Stand-Alone Executable
1. Download the Windows Executable from the ExifTool home page.
(The file you download should be named "exiftool-11.87.zip".)
2. Extract "exiftool(-k).exe" from the ".zip" file, and place it on your Desktop.
(Double-click on "exiftool-11.87.zip" to open the archive, then drag "exiftool(-k).exe" to your Desktop.)
3. You can now double-click on "exiftool(-k).exe" to read the application documentation, or drag-and-drop files and folders to run exiftool on selected files. To install exiftool for use from the command line, continue with the following steps:

4. Rename "exiftool(-k).exe" to "exiftool.exe".
(or "exiftool(-k)" to "exiftool" if file name extensions are hidden on your system)
5. Move "exiftool.exe" to the "C:\WINDOWS" directory (or any other directory in your PATH).

You can now run exiftool by typing "exiftool" at the command prompt. (To get to the command prompt, select "Run..." from the Windows "Start" menu, then type "cmd" and press Return.)
```

## Usage

This module can be used by importing it:

```python
import flir_image_extractor
fir = flir_image_extractor.FlirImageExtractor()
fir.process_image('examples/ax8.jpg')
fir.plot()
```

Or by calling it as a script:

```bash
python flir_image_extractor.py -p -i 'examples/zenmuse_xtr.jpg'
```

```bash
usage: flir_image_extractor.py [-h] -i INPUT [-p] [-exif EXIFTOOL]
                               [-csv EXTRACTCSV] [-d]

Extract and visualize Flir Image data

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input image. Ex. img.jpg
  -p, --plot            Generate a plot using matplotlib
  -exif EXIFTOOL, --exiftool EXIFTOOL
                        Custom path to exiftool
  -csv EXTRACTCSV, --extractcsv EXTRACTCSV
                        Export the thermal data per pixel encoded as csv file
  -d, --debug           Set the debug flag
```

This command will show an interactive plot of the thermal image using matplotlib and create two image files *flir_example_thermal.png* and *flir_example_rgb_image.jpg*. 
Both are RGB images, the original temperature array is available using the `get_thermal_np` or `export_thermal_to_csv` functions.

The functions `get_rgb_np` and `get_thermal_np` yield numpy arrays and can be called from your own script after importing this lib.

## Supported/Tested cameras:

- Flir One (thermal + RGB)
- Xenmuse XTR (thermal + thumbnail, set the subject distance to 1 meter)
- AX8 (thermal + RGB)

Other cameras might need some small tweaks (the embedded raw data can be in multiple image formats)

## Credits

Raw value to temperature conversion is ported from this R package: https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R
Original Python code from: https://github.com/Nervengift/read_thermal.py
