import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
# loading template images : using address of files to load all pattern images 
# we also turn their colors to gray for makeing sure it would be fine durring ahead process
def load(address):
    patterns = {}
    for name in sorted(os.listdir(address)):
            num = os.path.splitext(name)[0]
            path = os.path.join(address, name)
            image_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image_data is not None:
                patterns[num] = image_data
    return patterns

# now we want to take each picture of license plates and turn it apart to each character of license numbers , then we sort and merge it together for correlation process
def split_func(input, show_plots=True):
    if isinstance(input, str):
        real_image = cv2.imread(input) 
    else:
        real_image = input.copy() 
    # getting the size of image 
    h, w, _ = real_image.shape
    # we first change color to gray and then apply blur filter to improve quality 
    grayed = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayed, (5, 5), 0)
    # now we Apply an inverted binary threshold. This makes characters white and the background black.
    _, black_white = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # we  fill small gaps or holes and make them solid and easier to detect .
    filling_elements = np.ones((3,3), np.uint8)
    clean_image = cv2.morphologyEx(black_white, cv2.MORPH_CLOSE, filling_elements)
    # finding
    shapes, _ = cv2.findContours(clean_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    possible_shapes = []
    for shape in shapes:
        shape_area = cv2.contourArea(shape)
        # we  filter out shapes that are too small or too big (for canceling noise or errors ).
        if 50 < shape_area < (w * h / 5):
            possible_shapes.append(shape)
    if len(possible_shapes) >= 7: # our license has seven number but if system found less , it would keep it all
        possible_shapes.sort(key=cv2.contourArea, reverse=True)
        final_shapes = possible_shapes[:7]
    else:
        final_shapes = possible_shapes 
    # now we sort them and stick them together
    shape_list = sorted(final_shapes, key=lambda c: cv2.boundingRect(c)[0])
    final_list = []
    for char in shape_list:
        x, y, w, h = cv2.boundingRect(char)
        cropped_image = grayed[y:y+h, x:x+w]
        final_list.append(cropped_image)
    # here is the plot 
    if show_plots and final_list:
        plt.figure(figsize=(len(final_list) * 1.5, 2))
        if isinstance(input, str):
            plot_title = f'Segmented Characters from {os.path.basename(input)}'
        else:
            plot_title = "Segmented Characters from Image"
        plt.suptitle(plot_title)
        for i, character in enumerate(final_list):
            plt.subplot(1, len(final_list), i + 1)
            plt.imshow(character, cmap='gray')
            plt.axis('off')
        plt.show()
    # final list contain splitted and sortesd images which are needed for the next part
    return final_list

# in this part we have to indentify numbers or chars from the images , we first adapt some filters and size changes to make sure the comparision goes fair enough
# then we calculate correlation between desired image and our patterns 
# we choose the max of calculated correlation ratios (which is between -1 and 1) for best fitted image and that is our choice 
def recognize_func(char_list, patterns):
    #  getting a standard size 
    first = next(iter(patterns.values()))
    standard_h, standard_w = first.shape
    ans = ""
    correlations_ratios = []
    # now we loop through each segment to calcuate correlation
    for x in char_list:
        best = -1.0
        choice = ' '
        # first we prepare image in case of size 
        original_h, original_w = x.shape
        ratio = min(standard_w/original_w, standard_h/original_h)
        new_w, new_h = int(original_w*ratio), int(original_h*ratio)
        resized = cv2.resize(x, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # then we centreize the image for better handing
        padded = np.full((standard_h, standard_w), 255, dtype=np.uint8)
        x_start = (standard_w - new_w) // 2
        y_start = (standard_h - new_h) // 2
        padded[y_start:y_start+new_h, x_start:x_start+new_w] = resized
        # apply threshold to clean image from noises and make it more clear
        _, prepared = cv2.threshold(padded, 128, 255, cv2.THRESH_BINARY)
        # now we compare each segment with all patterns in this loop
        for name, image in patterns.items():
           # resize to avoid any mismatch
            template_resized = cv2.resize(image, (standard_w, standard_h))
            # calculate the score (correlation ratio which is between -1 and 1)
            match_result = cv2.matchTemplate(prepared, template_resized, cv2.TM_CCOEFF_NORMED)
            _, similarity_score, _, _ = cv2.minMaxLoc(match_result)
            # consider the best one through loop with choosing the max rate
            if similarity_score > best:
                best = similarity_score
                choice = name      
        # add the best matched one 
        ans += choice
        correlations_ratios.append(best)
    if len(ans) == 7: 
        ans = f"{ans[0:2]} {ans[2]} {ans[3:7]}"
    return ans, correlations_ratios

# main function : 
def main():
    # sources path
    script_dir = Path(__file__).parent
    letters_path = script_dir.parent / 'letters'
    numbers_path = script_dir.parent / 'numbers'
    plates_path = script_dir.parent / 'ideal'
    #loading sources
    number_samples = load(numbers_path)
    letter_samples = load(letters_path)
    samples = {**number_samples, **letter_samples}
    # at the end we want to verify if our system works or not so we save the correct answers  
    plates = ['p1.jpg', 'p2.jpg', 'p3.jpg', 'p4.jpg']
    correct_answers = {
        'p1.jpg': '98 C 7445', 'p2.jpg': '56 A 7495',
        'p3.jpg': '79 B 1208', 'p4.jpg': '93 D 4328'
    }
    # 4. Loop through each license plate file for testing.
    for plate_name in plates:
        plate_path = os.path.join(plates_path, plate_name)
        print(f"\nProcessing: {plate_name}")
        # first let's split the image and take it out chars which are needed
        chars = split_func(plate_path, show_plots=True)
        # calling the recognition function to identify chars
        output, rates = recognize_func(chars, samples)
        print(f"Result: {output}")
        print(f"Scores: {[f'{s:.2f}' for s in rates]}")
    # now we start downsampling , we want to check it out what is going on with our designed system through diffrent rates of downsampling 
    print("\n Starting Downsampling Analysis ")
    plate_down = 'p2.jpg'
    path_down = os.path.join(plates_path, plate_down)
    print(f"\nAnalyzing quality effects on: {plate_down}")
    original_plate_image = cv2.imread(path_down)
    correct_text = correct_answers[plate_down]
    # diffrent rates of down sampling : 
    rates = [1.0, 0.5, 0.25, 1/6, 1/8, 1/10]
    results = []
    # check the effect in diffrent rates through looping : 
    for rate in rates:
        h, w, _ = original_plate_image.shape
        low_quality_image = cv2.resize(original_plate_image, (int(w*rate), int(h*rate)), interpolation=cv2.INTER_AREA)
        # here is the plot
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(original_plate_image, cv2.COLOR_BGR2RGB)); plt.title(f'Original Image ({w}x{h})'); plt.axis('off')
        plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(low_quality_image, cv2.COLOR_BGR2RGB)); plt.title(f'Downsampled at {rate*100:.1f}% quality'); plt.axis('off')
        plt.suptitle(f"Downsampling Comparison (Rate: {rate:.3f})")
        plt.show()
        # now we want to run again the process of identification with low qaulity images 
        parts = split_func(low_quality_image, show_plots=False)
        text_result, rates_list = recognize_func(parts, samples)
        # checking the results
        check = (text_result == correct_text)
        average_rate = np.mean(rates_list) if rates_list else 0
        results.append((rate, text_result, check, average_rate))
    print("\nResults Table: Accuracy vs. Quality")
    print("Quality | Avg Score        | Correct/wrong | Recognized Text")
    print("--------------------------------------------------------------")
    # now we want to find min qaulity of images in which the system is working
    min_accepted_quality = None
    for rate, text, correct, score in sorted(results, key=lambda x: x[0], reverse=True):
        print(f"{rate:<7.3f} | {score:<16.4f} | {str(correct):<8}       |{text}")
        if correct: 
            min_accepted_quality = rate
    if min_accepted_quality: 
        print(f"\nConclusion: The system works down to approx. {min_accepted_quality:.3f} quality.")
if __name__ == "__main__":
    main()

