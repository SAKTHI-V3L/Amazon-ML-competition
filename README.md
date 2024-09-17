# Amazon-ML-competition
Feature extraction of measurements from product images like posters etc

TOPIC : FEATURE EXTRACTION FROM IMAGE
Project Overview:
This project aims to extract and standardize measurements from images using OCR (Optical Character Recognition). The measurements are extracted from images using predefined categories and regex patterns and then saved in a standardized format. The final output is stored in a CSV file with the highest measurement value for each image.


Key Components:

 1. Data Input & Setup:
    Input Data: A CSV file with columns index, image_link, and entity_name is loaded. The image_link contains URLs to the images, and entity_name specifies the measurement category (e.g., width, height, weight).
    Libraries: The code utilizes several Python libraries:
      pandas: For reading and writing CSV files.
      OpenCV (cv2): For loading and manipulating images.
      easyocr: For extracting text from the images.
      regex (re): For extracting specific measurements from the detected text.
      requests: To download images from URLs.

 2. Unit Patterns:
    Predefined Measurement Categories: The code uses regular expressions to match different measurement units like centimeters, feet, inches, etc. Each category (e.g., width, height, weight) has associated regex patterns that identify specific units.
    Categories include:
      Dimensions: width, height, depth
      Weight: item weight, maximum weight recommendation
      Electrical: voltage, wattage
      Volume: litres, gallons, etc.

 3. Standardization of Units:

Measurements extracted from the text are standardized to ensure uniformity across different units (e.g., converting feet to centimeters, inches to centimeters). This is done to simplify comparisons across different unit systems.
The standardize_units function ensures all values are formatted to a single decimal point, and a dictionary maps units to a standard form.


 Workflow:

1. Load CSV Data:
    The CSV file containing the image URLs is read, and each row is processed in sequence.

2. Fetch Image from URL:
    The image is fetched from the URL using requests and decoded using OpenCV to load it into memory.

3. Apply EasyOCR and Image Cropping:

The EasyOCR library is used to extract text from images, which helps detect and standardize measurements from the image content. In this script, the following techniques are employed to enhance OCR accuracy:
1.	Cropping the Image:
o	After OCR detects the bounding boxes around the detected text, the image is cropped to these bounding boxes using coordinates.
o	Cropped Image: By focusing on smaller sections of the image that contain text, you reduce noise and irrelevant background, improving OCR accuracy.
o	If the cropped image is empty (size = 0), the program skips further processing for that section to avoid errors.
2.	Image Resizing:
o	To enhance the recognition of small text, the cropped image is resized using OpenCV.
o	The resizing is done with a scaling factor (fx=2, fy=2), which enlarges the text region. Interpolation (INTER_LINEAR) is applied during resizing to smooth the pixels and retain clarity, thus improving OCR performance on small text.
3.	Reapplying OCR on Cropped Image:
o	Once the image is cropped and resized, EasyOCR is reapplied to these sections. This improves detection accuracy since smaller, resized regions are easier for the OCR engine to process, particularly when text is cluttered or small.
o	The OCR results from both the original and the cropped images are combined to ensure no information is missed.


4. Text Extraction and Pattern Matching:
    
After OCR detects text in the image, it is processed to extract specific measurements. 
The extract_measurements function scans the text using regex patterns defined for each entity category, such as width, height, or weight.
Regex Command Conditions:
The script uses regex (regular expressions) to extract measurements from text based on specific patterns. Each pattern captures numeric values followed by units (e.g., "cm", "kg", "inch"). Key components include:
•	\d+(\.\d+)?: Captures whole or decimal numbers.
•	\s*: Allows optional spaces between the number and the unit.
•	(?!\w): Ensures the unit is not followed by other letters to avoid incorrect matches. Patterns are tailored for each measurement category like width, height, weight, voltage, and volume, covering all possible unit variations (e.g., "cm", "centimetres").


5. Unit Conversion and Standardization:
    Measurements are standardized to a common unit using the standardize_units function. For example, feet are converted to centimeters, and pounds to kilograms.

6. Find the Highest Measurement:
    The highest measurement for each category is identified using the find_highest_measurement function. This helps in ensuring that the most significant value is considered for storage.

7. Save Results:
    The highest value for each image is saved, and the results are written to a new CSV file. Each entry in the output file includes the image index and its corresponding highest measurement value.


Key Functions:

1. url_to_image:
    This function converts an image URL to an OpenCVreadable image. It uses the requests library to download the image, decodes it, and loads it into memory for further processing.

2. extract_measurements:
    This function scans the text extracted from OCR and matches patterns based on the measurement category (width, height, etc.). Regular expressions for each category are used to identify units and extract values.

3. standardize_units:
    Converts various units (e.g., feet, inches) to a common standardized form for easier comparison.

4. find_highest_measurement:
    This function processes the extracted measurements, converts them to a base unit, and identifies the highest value. The result is the most significant measurement for each image.

5. convert_to_base_unit:
    Converts each measurement into a base unit using predefined conversion factors (e.g., feet to centimeters).

Output:
 Final CSV File: The code outputs a CSV file with two columns:
    index: Corresponding to the image’s original index.
    entity_value: The highest measurement extracted from the image.

Summary:

This project automates the extraction and standardization of measurements from images using OCR and regex. The workflow processes images, extracts measurements based on category, standardizes units, and stores the highest value in a CSV file. This approach can be extended to various use cases, such as cataloging product dimensions, analyzing technical specifications, or processing inventory data.
