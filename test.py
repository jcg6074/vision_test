import cv2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
data_dir = "covid"

# 데이터 변환 설정 (예: 리사이즈, 텐서 변환, 정규화 등)
data_transforms = transforms.Compose([
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

# 데이터를 ImageFolder 형식으로 불러오기
dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)

# 데이터셋 크기 확인
dataset_size = len(dataset)
print(f"Total number of images: {dataset_size}")

# Train-Test split 비율 설정
train_ratio = 0.8
train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size

# 데이터셋 나누기
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoader 설정
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 클래스 확인
class_names = dataset.classes