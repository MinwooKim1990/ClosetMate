import os
import pandas as pd

def debug_style_items(image_paths, predicted_labels):
    style_details = {}
    
    for path, label in zip(image_paths, predicted_labels):
        filename = os.path.basename(path)
        style_number = int(filename.split('-')[0])
        
        if style_number not in style_details:
            style_details[style_number] = []
            
        style_details[style_number].append({
            'filename': filename,
            'predicted_label': label
        })
    
    print("\n=== 스타일별 상세 정보 ===")
    for style, details in sorted(style_details.items()):
        print(f"\n스타일 {style}:")
        print(f"총 이미지 개수: {len(details)}")
        print("이미지 목록:")
        for item in details:
            print(f"  파일명: {item['filename']}, 예측 라벨: {item['predicted_label']}")

def create_style_clothing_matrix(image_paths, predicted_labels, clothing_dict):
    style_numbers = []
    style_items = {}
    style_files = {} 
    
    print("\n=== 데이터 처리 과정 ===")
    for path, label in zip(image_paths, predicted_labels):
        filename = os.path.basename(path)
        style_number = int(filename.split('-')[0])
        style_numbers.append(style_number)
        
        if style_number not in style_items:
            style_items[style_number] = set()
            style_files[style_number] = {}
        
        style_items[style_number].add(label)
        
        if label not in style_files[style_number]:
            style_files[style_number][label] = []
        style_files[style_number][label].append(filename)
        
        print(f"처리 중: {filename} -> 스타일 {style_number}, 라벨: {label}")
    
    unique_styles = sorted(set(style_numbers))
    
    clothing_types = list(dict.fromkeys(clothing_dict.values()))
    
    data = []
    
    print("\n=== 스타일별 의류 종류 ===")
    for style in unique_styles:
        clothing_presence = {'style': style}
        
        print(f"\n스타일 {style}의 의류:")
        
        for clothing_type in clothing_types:
            if clothing_type in style_items[style]:
                clothing_presence[clothing_type] = ', '.join(style_files[style][clothing_type])
                print(f"  - {clothing_type}: {clothing_presence[clothing_type]}")
            else:
                clothing_presence[clothing_type] = ''
        
        data.append(clothing_presence)
    
    df = pd.DataFrame(data)
    
    columns = ['style'] + clothing_types
    df = df[columns]
    
    return df

# get random style for selected brand
def get_random_style_images(csv_path, image_folder):
    df = pd.read_csv(csv_path)
    
    random_style_row = df.sample(n=1).iloc[0]
    style_number = int(random_style_row['style'])
    
    image_files = []
    for column in df.columns[1:]:
        value = random_style_row[column]
        if pd.notna(value) and str(value).strip():
            files = str(value).split(', ')
            image_files.extend(files)
    
    image_paths = [os.path.join(image_folder, filename) for filename in image_files]
    
    print(f"선택된 스타일: {style_number}")
    print(f"포함된 의류 종류:")
    cat=[]
    for column in df.columns[1:]:
        value = random_style_row[column]
        if pd.notna(value) and str(value).strip():
            cat.append(column)
            print(f"- {column}: {value}")
    
    return style_number, image_paths, cat

def categorize_item(item_name):
    top_categories = {'Jacket', 'Coat', 'Jumper', 'Cardigan', 'Vest'}
    inner_top_categories = {'Knit', 'Sweater', 'Shirt', 'Blouse'}
    bottom_categories = {'Skirt', 'Pants'}
    shoes_categories = {'Shoes'}
    accessory_categories = {'Bag', 'Scarf', 'Hat & Cap', 'One Piece'}

    if item_name in top_categories:
        return "Cover Upper"
    elif item_name in inner_top_categories:
        return "Inner Upper"
    elif item_name in bottom_categories:
        return "Under"
    elif item_name in shoes_categories:
        return "Shoes"
    elif item_name in accessory_categories:
        return "Acc"
    else:
        return "None"
