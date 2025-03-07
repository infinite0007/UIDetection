# UIDetection Scientific Project
Framework that leverages modern computer vision technologies to analyze individual image fragments, recognize the system’s operational mode, and verify that all elements are correctly positioned. This framework will serve as the backbone for automated testing, enabling reliable and efficient validation of visual elements in various test scenarios.

## Installation Steps
1. Go sure you made the installation simple by using python 3.12.x otherwhise errors could appear and extra steps are necessary. On our build systems we had 7.0.0 and that worked with the additional steps in 4 also but as I mentioned is more work to do.

2. Install all necessary dependencies with:<br>
`pip install -r requirements.txt` if your python version is under 9.0 and/or an error appears on installing the requirements.txt (Cargo Rust Package Manager) jump to step. 4

2. If You have Internet skip this step! Otherwhise the OCR-Ai-model will not be automatically pulled. You have to put the EasyOCR into the User folder in Windows. For That copy the .EasyOCR folder and copy it to: C:\Users\username\
The path should then look like that: C:\Users\lhglij1\\.EasyOCR

3. Run script with: `python ui_detection.py`

4. Only do this if error on step 1. appear. Make sure you have Visual Studio installed with the extension: C++ Desktop developement with MSVC. This is needed for the following Rust distribution to prevent errors on compile. Watch the installation guide here: https://rust-lang.github.io/rustup/installation/windows-msvc.html
Download the Rust-package manager .exe on: https://rustup.rs/ or if no internet download the offline-installer msvc (same 32/64bit version as your python interpreter) on https://forge.rust-lang.org/infra/other-installation-methods.html. Choose option 1. for standard installation and install the option for the current user with path-variables. for installing Visual Studio Desktop development with C++. Also when starting the Visual Studio installer make sure that package was installed successfully.
<br><br>
Try step. 1 again if problems continue here (normally with python versions 7.x.x):
<br><br>
OrderedDict import error: Go to pythonxx/lib/typing.py open and add following import on top:<br>`from collections import OrderedDict` 
<br><br>
urllib3 not found error (run following in terminal):
<br>
`pip uninstall urllib3`<br>
`pip install urllib3==1.26.7`<br>
<br>
start from step. 1 again



## Requirements from University
- critical view on project mandatory
- why, how it is done
- comparison to other papers and state of the art approaches
- scientific research needed
- ca 3-6 month

## Requirements from Liebherr
#### Non functional:
- runnable on standard lhg notebook
- input full hd image or video
- image brightness might be not optimal
- angle might be not optimal
- callable from python and integrateable in automation systems like jenkins
- processing of single image shall not exceed 10s
- framework shall be adaptable for different screens and sizes
- Why we have a analogl approach for a digital solution? (Freezer Display → Camera → Code)
- is it worth to have a more expensive equipment or is a casual camera enough? Are there any advantages in long-term?

#### functional must have:
- position and size of display shall be detectable
- analyse output shall include number of pixel per color
- analyse output shall include histogram pro color
- detection of string content shall be possible (ocr)
- it shall be possible to check if a given image is part of screen and if position is correct
- detection of animation like blinking elements shall be possible
  - output shall contain at least number of changing pixel, frequency and duty cycle

#### functional mid term:
- automated interface to UI-sim
  - to check content of screen against UI-sim
  - to check flow against UI-sim
- detection of ghosting elements