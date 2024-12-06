# UIDetection Scientific Project
Framework that leverages modern computer vision technologies to analyze individual image fragments, recognize the system’s operational mode, and verify that all elements are correctly positioned. This framework will serve as the backbone for automated testing, enabling reliable and efficient validation of visual elements in various test scenarios.

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