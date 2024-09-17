import pandas as pd
import cv2
import easyocr
import re
import requests
import numpy as np
from io import BytesIO
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Define unit patterns for each category with updated regex patterns
unit_patterns = {
    "width": {
        'centimetre': r'(\d+(\.\d+)?)\s*(centimetre|centimetres|cm)(?!\w)',
        'foot': r'(\d+(\.\d+)?)\s*(foot|feet|ft|\'|”)(?!\w)',
        'millimetre': r'(\d+(\.\d+)?)\s*(millimetre|millimetres|mm)(?!\w)',
        'metre': r'(\d+(\.\d+)?)\s*(metre|metres|m)(?!\w)',
        'inch': r'(\d+(\.\d+)?)\s*(inch|inches|in|”)(?!\w)',
        'yard': r'(\d+(\.\d+)?)\s*(yard|yards|yd)(?!\w)'
    },
    "height": {
        'centimetre': r'(\d+(\.\d+)?)\s*(centimetre|centimetres|cm)(?!\w)',
        'foot': r'(\d+(\.\d+)?)\s*(foot|feet|ft|\'|”)(?!\w)',
        'millimetre': r'(\d+(\.\d+)?)\s*(millimetre|millimetres|mm)(?!\w)',
        'metre': r'(\d+(\.\d+)?)\s*(metre|metres|m)(?!\w)',
        'inch': r'(\d+(\.\d+)?)\s*(inch|inches|in|”)(?!\w)',
        'yard': r'(\d+(\.\d+)?)\s*(yard|yards|yd)(?!\w)'
    },
    "depth": {
        'centimetre': r'(\d+(\.\d+)?)\s*(centimetre|centimetres|cm)(?!\w)',
        'foot': r'(\d+(\.\d+)?)\s*(foot|feet|ft|\'|”)(?!\w)',
        'millimetre': r'(\d+(\.\d+)?)\s*(millimetre|millimetres|mm)(?!\w)',
        'metre': r'(\d+(\.\d+)?)\s*(metre|metres|m)(?!\w)',
        'inch': r'(\d+(\.\d+)?)\s*(inch|inches|in|”)(?!\w)',
        'yard': r'(\d+(\.\d+)?)\s*(yard|yards|yd)(?!\w)'
    },
    "item_weight": {
        'milligram': r'(\d+(\.\d+)?)\s*(milligram|milligrams|mg)(?!\w)',
        'kilogram': r'(\d+(\.\d+)?)\s*(kilogram|kilograms|kg)(?!\w)',
        'microgram': r'(\d+(\.\d+)?)\s*(microgram|micrograms|µg)(?!\w)',
        'gram': r'(\d+(\.\d+)?)\s*(gram|grams|g)(?!\w)',
        'ounce': r'(\d+(\.\d+)?)\s*(ounce|ounces|oz)(?!\w)',
        'ton': r'(\d+(\.\d+)?)\s*(ton|tons|t)(?!\w)',
        'pound': r'(\d+(\.\d+)?)\s*(pound|pounds|lb|lbs)(?!\w)'
    },
    "maximum_weight_recommendation": {
        'milligram': r'(\d+(\.\d+)?)\s*(milligram|milligrams|mg)(?!\w)',
        'kilogram': r'(\d+(\.\d+)?)\s*(kilogram|kilograms|kg)(?!\w)',
        'microgram': r'(\d+(\.\d+)?)\s*(microgram|micrograms|µg)(?!\w)',
        'gram': r'(\d+(\.\d+)?)\s*(gram|grams|g)(?!\w)',
        'ounce': r'(\d+(\.\d+)?)\s*(ounce|ounces|oz)(?!\w)',
        'ton': r'(\d+(\.\d+)?)\s*(ton|tons|t)(?!\w)',
        'pound': r'(\d+(\.\d+)?)\s*(pound|pounds|lb|lbs)(?!\w)'
    },
    "voltage": {
        'millivolt': r'(\d+(\.\d+)?)\s*(millivolt|millivolts|mV)(?!\w)',
        'kilovolt': r'(\d+(\.\d+)?)\s*(kilovolt|kilovolts|kV)(?!\w)',
        'volt': r'(\d+(\.\d+)?)\s*(volt|volts|V)(?!\w)'
    },
    "wattage": {
        'kilowatt': r'(\d+(\.\d+)?)\s*(kilowatt|kilowatts|kW)(?!\w)',
        'watt': r'(\d+(\.\d+)?)\s*(watt|watts|W)(?!\w)'
    },
    "volume": {
        'cubic foot': r'(\d+(\.\d+)?)\s*(cubic foot|cubic feet|cu ft|ft³)(?!\w)',
        'microlitre': r'(\d+(\.\d+)?)\s*(microlitre|microlitres|µL)(?!\w)',
        'cup': r'(\d+(\.\d+)?)\s*(cup|cups)(?!\w)',
        'fluid ounce': r'(\d+(\.\d+)?)\s*(fluid ounce|fluid ounces|fl oz)(?!\w)',
        'centilitre': r'(\d+(\.\d+)?)\s*(centilitre|centilitres|cl)(?!\w)',
        'imperial gallon': r'(\d+(\.\d+)?)\s*(imperial gallon|imperial gallons|imp gal)(?!\w)',
        'pint': r'(\d+(\.\d+)?)\s*(pint|pints|pt)(?!\w)',
        'decilitre': r'(\d+(\.\d+)?)\s*(decilitre|decilitres|dl)(?!\w)',
        'litre': r'(\d+(\.\d+)?)\s*(litre|litres|L)(?!\w)',
        'millilitre': r'(\d+(\.\d+)?)\s*(millilitre|millilitres|ml)(?!\w)',
        'quart': r'(\d+(\.\d+)?)\s*(quart|quarts|qt)(?!\w)',
        'cubic inch': r'(\d+(\.\d+)?)\s*(cubic inch|cubic inches|cu in|in³)(?!\w)',
        'gallon': r'(\d+(\.\d+)?)\s*(gallon|gallons|gal)(?!\w)'
    }
}


# Conversion functions to standardize units
def standardize_units(value, unit):
    # Format the value to always have a decimal point
    value = f"{float(value):.1f}"
    conversion = {
        'centimetre': 'centimetre',
        'foot': 'foot',
        'feet': 'foot',  # Convert 'feet' to 'foot'
        'millimetre': 'millimetre',
        'metre': 'metre',
        'inch': 'inch',
        'yard': 'yard',
        'milligram': 'milligram',
        'kilogram': 'kilogram',
        'microgram': 'microgram',
        'gram': 'gram',
        'ounce': 'ounce',
        'ton': 'ton',
        'pound': 'pound',
        'millivolt': 'millivolt',
        'kilovolt': 'kilovolt',
        'volt': 'volt',
        'cubic foot': 'cubic foot',
        'microlitre': 'microlitre',
        'cup': 'cup',
        'fluid ounce': 'fluid ounce',
        'centilitre': 'centilitre',
        'imperial gallon': 'imperial gallon',
        'pint': 'pint',
        'decilitre': 'decilitre',
        'litre': 'litre',
        'millilitre': 'millilitre',
        'quart': 'quart',
        'cubic inch': 'cubic inch',
        'gallon': 'gallon'
    }
    standard_unit = conversion.get(unit, unit)
    return f"{value} {standard_unit}"

# Function to extract and standardize measurements
def extract_measurements(text, category):
    results = []
    patterns = unit_patterns.get(category, {})

    if not patterns:
        print(f"Category '{category}' is not valid.")
        return results

    text = ' '.join(text)  # Join the list into a single string
    for unit, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            value = match[0]
            standardized_measurement = standardize_units(value, unit)
            results.append(standardized_measurement)

    return results

# Function to convert image URL to OpenCV image
def url_to_image(url):
    response = requests.get(url)
    image = np.array(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

# Conversion factors for base unit calculations
conversion_factors = {
    'centimetre': 1,
    'millimetre': 0.1,
    'metre': 100,
    'foot': 30.48,
    'inch': 2.54,
    'yard': 91.44,
    'milligram': 0.001,
    'gram': 1,
    'kilogram': 1000,
    'microgram': 1e-6,
    'ounce': 28.3495,
    'pound': 453.592,
    'ton': 1e6,
    'millivolt': 0.001,
    'volt': 1,
    'kilovolt': 1000,
    'watt': 1,
    'kilowatt': 1000,
    'litre': 1,
    'millilitre': 0.001,
    'gallon': 3.78541,
    'pint': 0.473176,
    'cup': 0.24,
    'fluid ounce': 0.0295735,
    'cubic foot': 28.3168,
    'cubic inch': 0.0163871,
    'imperial gallon': 4.54609
}

# Helper function to convert a measurement to its base unit
def convert_to_base_unit(value, unit):
    base_value = float(value) * conversion_factors.get(unit, 1)
    return base_value

# Find the highest measurement
def find_highest_measurement(measurements):
    highest_value = None
    highest_measurement = None
    for measurement in measurements:
        value, unit = measurement.split(' ', 1)
        base_value = convert_to_base_unit(value, unit)
        # Check if it's the highest value and update the highest value
        if highest_value is None or base_value > highest_value:
            highest_value = base_value
            highest_measurement = measurement

    return highest_measurement

# Main code to process input and output
def process_file(input_file, output_file):
    df = pd.read_csv(input_file)

    results = []
    reader = easyocr.Reader(['en'])

    for index, row in df.iterrows():
        image_url = row['image_link']
        entity_name = row['entity_name']

        # Convert URL to image and process with OCR
        image = url_to_image(image_url)
        detection_results = reader.readtext(image)

        # List to store all detected text
        all_detected_text = []

        # Iterate through the detected text and their bounding boxes
        for (bbox, text, confidence) in detection_results:
            # Add the detected text to the list
            all_detected_text.append(text)

            # Extract the coordinates of the bounding box
            top_left = (int(bbox[0][0]), int(bbox[0][1]))
            bottom_right = (int(bbox[2][0]), int(bbox[2][1]))

            # Crop the image to the bounding box
            cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            # Check if the cropped image is not empty
            if cropped_image.size == 0:
                print("Cropped image is empty. Skipping this section.")
                continue

            # Optional: Resize the cropped image for better OCR accuracy
            cropped_image_resized = cv2.resize(cropped_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

            # Re-apply OCR on the cropped image
            results_cropped = reader.readtext(cropped_image_resized)
            for (_, cropped_text, _) in results_cropped:
                all_detected_text.append(cropped_text)

        # Extract measurements based on the category
        extracted_measurements = extract_measurements(all_detected_text, entity_name)
        highest_measurement = find_highest_measurement(extracted_measurements) if extracted_measurements else ''

        results.append([index, highest_measurement])

    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results, columns=['index', 'entity_value'])
    results_df.to_csv(output_file, index=False)


# Usage
input_csv_file = '/content/sample_test.csv'
output_csv_file = 'output_file.csv'
process_file(input_csv_file, output_csv_file)