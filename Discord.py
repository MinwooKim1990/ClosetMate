#bot_new_2-2를 기준으로 진행 + 추천코디 완(숫자로만 입력받게) + 그라디오 연결 : 완!!!!

import discord
from discord.ext import commands
from dotenv import load_dotenv
import torch
import numpy as np
import pandas as pd
import os
import io
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
import warnings
from tools.Extractor import FeatureExtractor
from tools.ExtractFeature import extract_features_from_images
from tools.Fashiondataset import FashionDataset
from tools.clustering import perform_clustering_with_cache, get_cluster_categories
from tools.similarity2 import Similarity

warnings.filterwarnings("ignore", category=UserWarning)

# 환경 설정
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 랜덤 시드 설정
torch.manual_seed(2024)
np.random.seed(2024)

# 전역 설정 및 변수 초기화
user_name = 'default'
model_type = 'deit'  # default_bagged_deit_feature.npy 사용을 위한 모델 타입 설정
no_clusters = 21
image_folder = 'C:/MS_AI_School_Rim/project/ClosetMate/data/img_jpg'
metadata_path = 'C:/MS_AI_School_Rim/project/ClosetMate/data/metadata_with_Nan.csv'
feature_file_path = 'C:/MS_AI_School_Rim/project/ClosetMate/saved/default_bagged_deit_feature.npy'

# 1. 데이터 로드 및 클러스터링 모델 설정
df = pd.read_csv(metadata_path)
names = pd.DataFrame(set(df['Name']), columns=['Name'])
Dataset = FashionDataset(dataframe=names, image_folder=image_folder)
Dataloader = DataLoader(Dataset, batch_size=16, shuffle=False)

# 특징 파일 로드 또는 생성 (default_bagged_deit_feature.npy 사용)
if os.path.exists(feature_file_path):
    features = np.load(feature_file_path)
    print(f"✔️ '{feature_file_path}' 파일에서 특징을 로드했습니다.")
else:
    print("🚨 특징 파일이 존재하지 않습니다. 새로 추출이 필요합니다.")

## 카테고리명 통합 (수정)
from tools.module import enhance_features_with_category_weights_optimized
en_feature = enhance_features_with_category_weights_optimized(features, Dataset)

# 클러스터링 모델 로드 또는 생성
labels, kmeans = perform_clustering_with_cache(user_name, en_feature, model_type=model_type, n_clusters=no_clusters, force_cluster=False)
print("✔️ 클러스터링 모델과 라벨이 로드되었습니다.")

# 카테고리 매핑 정보
category_mapping = {
    'JK': 'Jacket', 'CT': 'Coat', 'JP': 'Jumper', 'KN': 'Knit', 'SW': 'Sweater',
    'SH': 'Shirt', 'BL': 'Blouse', 'CD': 'Cardigan', 'VT': 'Vest', 'OP': 'One Piece',
    'SK': 'Skirt', 'PT': 'Pants', 'SE': 'Shoes', 'BG': 'Bag', 'SC': 'Scarf', 'HC': 'Hat & Cap'
}

# 폴더 내 이미지 경로 추출 함수
def get_image_paths(folder_path, recursive=False):
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {folder_path}")
    pattern = "**/*" if recursive else "*"
    return sorted(str(file_path) for file_path in folder.glob(pattern) if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS)

# 여러 이미지를 결합하는 함수
def combine_images(images, rows=1, cols=5, target_size=(200, 200)):
    # 각 이미지를 지정된 크기로 리사이즈
    resized_images = [img.resize(target_size, Image.LANCZOS) for img in images if img is not None]
    width, height = target_size

    # 결합할 빈 캔버스 생성
    combined_image = Image.new('RGB', (width * cols, height * rows), (255, 255, 255))
    
    # 이미지를 캔버스에 붙여 넣기
    for idx, img in enumerate(resized_images):
        row, col = divmod(idx, cols)
        combined_image.paste(img, (col * width, row * height))
    
    return combined_image

# 2. 이미지 업로드 및 유사도 기반 추천

@bot.command(name="옷짱찾기")
async def recommend(ctx):
    if not ctx.message.attachments:
        await ctx.send("✔️ 추천을 위해 이미지를 업로드해 주세요.")
        return
    
    # 업로드된 이미지 저장 경로 설정
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    attachment = ctx.message.attachments[0]
    img_path = os.path.join('uploads', attachment.filename)
    await attachment.save(img_path)
    await ctx.send(f"▪️▫️▫️▫️ (🧥25%) \n이미지를 저장하고 특징을 추출 중입니다...")

    # 특징 추출
    feature_extractor = FeatureExtractor(model_type=model_type).to(device)
    image_features = extract_features_from_images([img_path], feature_extractor, device)
    
    # 클러스터 예측
    predicted_cluster = kmeans.predict(image_features)[0]

    # 군집 카테고리 정보 제공
    cluster_categories = get_cluster_categories(Dataset, kmeans, en_feature, labels, threshold=0.0)
    cluster_name = category_mapping.get(cluster_categories.get(predicted_cluster, ''), 'Unknown')
    await ctx.send(f"▪️▪️▫▫️ (🧥50%) \n이 이미지는 **군집 {predicted_cluster} ({cluster_name})**에 속합니다.")

    # Similarity 객체 생성 및 유사도 기반 추천
    sim = Similarity(
        feature_vectors=en_feature,
        dataset=Dataset,
        user_name=user_name,
        user_vector=image_features,
        model_type=model_type,
        force_new=False
    )
    
    # 상위 5개의 유사한 아이템 가져오기
    similar_items = sim.get_sim(image_features, k=5)[0]
    
    # 구매 매력도 검사
    percentages, recommendation = sim.attractiveness(image_features, return_percentage=True, custom_threshold=None)

    # 추천 결과 및 유사도 출력
    await ctx.send(f"▪️▪️▪️▫ (🧥75%) \n추천 결과: \n**{recommendation}**")
    await ctx.send("▪️▪️▪️▪️ (🧥100%) \n상위 5개 아이템 유사도:")
    for i, percent in enumerate(percentages[0][:5], 1):
        await ctx.send(f"{i}번째 아이템: {percent:.2f}%")

    # 이미지 표시
    images = []
    for idx in similar_items:
        if idx < len(Dataset.dataframe):
            img_name = Dataset.dataframe.iloc[idx]['Name']
            img_path_jpg = os.path.join(image_folder, f"{img_name}.jpg")
            
            if os.path.exists(img_path_jpg):
                try:
                    img = Image.open(img_path_jpg).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"이미지 로드 오류: {img_path_jpg}. 오류: {e}")
            else:
                print(f"파일이 존재하지 않습니다: {img_path_jpg}")
        else:
            print(f"Invalid index {idx} for DataFrame of length {len(Dataset.dataframe)}")

    # 추천된 이미지가 5개 이하일 경우 빈 이미지를 추가합니다.
    while len(images) < 5:
        images.append(Image.new('RGB', (200, 200), (255, 255, 255)))  # 빈 이미지

    # 여러 이미지를 하나로 결합하여 전송
    if images:
        combined_image = combine_images(images, rows=1, cols=5)
        with io.BytesIO() as image_binary:
            combined_image.save(image_binary, 'PNG')
            image_binary.seek(0)
            await ctx.send(file=discord.File(fp=image_binary, filename="recommendations.png"))
    else:
        await ctx.send("추천할 이미지가 없습니다.")


from tools.recommend import debug_style_items, create_style_clothing_matrix, get_random_style_images, categorize_item  # import helper functions


##################################################################################################

# 클러스터링 모델 로드 또는 생성
labels, kmeans = perform_clustering_with_cache(user_name, en_feature, model_type=model_type, n_clusters=no_clusters, force_cluster=False)
print("✔️ 클러스터링 모델과 라벨이 로드되었습니다.")

# cluster_names 설정: 클러스터 번호와 카테고리 이름 매핑
from tools.clustering import get_cluster_categories

cluster_categories = get_cluster_categories(
    Dataset, 
    kmeans, 
    en_feature, 
    labels,
    threshold=0.0
)

# 클러스터 번호와 카테고리 이름 연결
cluster_names = {k: category_mapping.get(v, 'Unknown') for k, v in cluster_categories.items()}
print(cluster_names) 

@bot.command(name="옷짱추천")
async def style_recommendation(ctx):
    # 스타일 옵션 및 메시지 출력
    style_options = {
        "1": "Athlezer",
        "2": "Casual",
        "3": "Office",
        "4": "Feminine"
    }
    await ctx.send(
        "아래 스타일 중 하나를 선택하세요:\n"
        "1️⃣ **애슬래저**\n"
        "2️⃣ **캐주얼**\n"
        "3️⃣ **오피스**\n"
        "4️⃣ **페미닌**"
    )

    # 사용자 입력 확인 함수
    def check(m):
        return m.author == ctx.author and m.content in style_options.keys()

    # 사용자 스타일 선택 입력 받기
    try:
        msg = await bot.wait_for("message", check=check, timeout=30)
        path = style_options[msg.content]
        await ctx.send(f"**{path} 스타일**이 선택되었습니다.")
    except:
        await ctx.send("시간 초과되었습니다. 다시 시도해주세요.")
        return

    # 스타일 이미지 경로 및 랜덤 이미지 선택
    img_path = f"C:/MS_AI_School_Rim/project/ClosetMate/data/{path}"
    image_paths = get_image_paths(img_path)

    if not image_paths:
        await ctx.send(f"{path} 스타일의 이미지 파일을 찾을 수 없습니다.")
        return

    try:
        # 랜덤으로 추천 이미지 및 스타일 정보 추출
        style_num, selected_image_paths, categories = get_random_style_images(f'{img_path}/clothing_matrix_{path}.csv', img_path)
        await ctx.send(f"추천된 스타일 번호: **{style_num}번**")
    except Exception as e:
        await ctx.send(f"오류 발생: {str(e)}")
        return

    # 추천된 이미지 파일 존재 확인 및 추가
    images = []
    for img_path in selected_image_paths[:5]:
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"이미지 로드 오류: {img_path}, 오류: {e}")
        else:
            print(f"파일이 존재하지 않습니다: {img_path}")

    while len(images) < 5:  # 5개 미만인 경우 빈 이미지 추가
        images.append(Image.new('RGB', (200, 200), (255, 255, 255)))

    # 이미지 결합 및 전송
    combined_image = combine_images(images, rows=1, cols=5)
    with io.BytesIO() as image_binary:
        combined_image.save(image_binary, 'PNG')
        image_binary.seek(0)
        await ctx.send(file=discord.File(fp=image_binary, filename="style_recommendations.png"))

###############################################################################################
# Gradio 앱 정의

gradio_url = "https://2936ee681d5eb29a18.gradio.live/"

@bot.command(name="옷짱열기")
async def gradio(ctx):
    await ctx.send(f"🌐 내 옷짱에 접속하기: \n{gradio_url}")

# 봇 실행
bot.run(TOKEN)

