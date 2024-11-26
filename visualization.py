def plot_attention_map_on_image(attn_weights, original_img, patch_size=8, img_size=32):
    """
    Attention map을 원본 이미지에 겹쳐서 시각화하는 함수
    """
    num_heads, num_patches, _ = attn_weights.shape
    attn_weights = attn_weights.mean(dim=0).cpu().numpy()  # 여러 heads의 평균을 구함
    
    # Attention Map을 (patch_size, patch_size) 형태로 재구성
    attn_map = attn_weights.reshape(int(img_size / patch_size), int(img_size / patch_size), -1)
    
    # Attention Map을 0-255 범위로 스케일링
    attn_map_resized = cv2.resize(attn_map, (original_img.shape[1], original_img.shape[0]))
    attn_map_resized = np.uint8(255 * attn_map_resized)  # 0-255 범위로 변환
    
    # 컬러 맵을 사용해 Attention Map을 색상 맵으로 변환
    heatmap = cv2.applyColorMap(attn_map_resized, cv2.COLORMAP_JET)

    # 원본 이미지에 Heatmap을 겹침
    overlay = cv2.addWeighted(original_img, 0.7, heatmap, 0.3, 0)

    # 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.title("Attention Map Overlaid on Image")
    plt.axis('off')
    plt.show()

# 원본 이미지 불러오기
original_img = cv2.imread('test_image.jpg')  # 실제 이미지 경로로 변경
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR로 읽으므로 RGB로 변환

# Test 데이터에 대해 Attention Map을 원본 이미지에 겹쳐서 시각화
plot_attention_map_on_image(attention_weights[-1], original_img)