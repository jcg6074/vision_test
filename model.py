import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class layer_norm(nn.Module):
    def __init__(self, patch, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm([patch, hidden_dim])
    def forward(self, x):
        x = self.norm(x)
        return x

class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.Lelu()

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x
        
class cross_attention(nn.Module):
    def __init__(self, head = 4, dim1, dim2):
        super().__init__()
        self.q = nn.Linear(dim2, dim2)
        self.K = nn.Linear(dim1, dim2)
        self.V = nn.Linear(dim1, dim2)
        self.u = nn.Linear(dim2, dim2)
        self.softmax = nn.Softmax(-1)
        self.head = head
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x1, x2):
        B1, P1, C1 = x1.shape
        B2, P2, C2 = x2.shape
        
        q = self.q(x2).reshape(B2, P2, self.num_head, -1).transpose(1, 2)
        k = self.k(x1).reshape(B1, P1, self.num_head, -1).transpose(1, 2)
        v = self.v(x1).reshape(B2, P1, self.num_head, -1).transpose(1, 2)
        
        attn = q @ k.transpose(-1, -2) / np.sqrt(C2)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B2, P2, C2)
        x = self.u(x)
        return x
    

class self_attention(nn.Module):
    def __init__(self, head = 4, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.u = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(-1)
        self.head = head
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        B, P, C = x.shape
        
        q = self.q(x).reshape(B, P, self.num_head, -1).transpose(1, 2)
        k = self.k(x).reshape(B, P, self.num_head, -1).transpose(1, 2)
        v = self.v(x).reshape(B, P, self.num_head, -1).transpose(1, 2)
        
        attn = q @ k.transpose(-1, -2) / np.sqrt(C)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, P, C)
        x = self.u(x)
        return x, attn

class att(nn.Module):
    def __init__(self, head=4, patch, dim):
        super().__init__()
        self.attention = self_attention(head)
        self.layer_norm1 = layer_norm(patch, dim)
        self.layer_norm2 = layer_norm(patch, dim)
        self.mlp = MLP()
    
    def forward(self,x):
        h = self.layer_norm1(x)
        sa, attn = self.attention(h)
        x = x + sa

        h = self.layer_norm2(x)
        ma = self.mlp(h)
        x = x + ma

        return x, attn

class Cross_Fusion(nn.Module):
    def __init__(self, head=4, patch1, patch2, dim1, dim2):
        super().__init__()
        self.cross_attention = cross_attention(head)
        self.layer_norm1 = layer_norm(patch1, dim1)
        self.layer_norm2 = layer_norm(patch2, dim2)
        self.layer_norm3 = layer_norm(patch2, dim2)
        self.mlp = MLP()

    def forward(self, x1, x2):
        h1 = self.layer_norm1(x1)
        h2 = self.layer_norm2(x2)
        ca = self.cross_attention(h1,h2)
        x = x2+ca

        h = self.layer_norm3(x)
        ma = self.mlp(h)
        x = x+ma

        return x


class STEM(nn.Module):
    def __init__(self, input_dim, dim1, dim2, dim3, len1, len2, len3):
        super().__init__()
        self.conv1 = nn.Conv2D(input_dim, dim1, (4,4), (4,4))
        self.conv2 = nn.Conv2D(dim1, dim2, (4,4), (4,4))
        self.conv3 = nn.Conv2D(dim2, dim3, (4,4), (4,4))

        self.PE1 = torch.nn.Parameter(torch.zeros(1, dim1, len1, len1))
        self.PE2 = torch.nn.Parameter(torch.zeros(1, dim2, len2, len2))
        self.PE3 = torch.nn.Parameter(torch.zeros(1, dim3, len3, len3))
 

    def forward(self, x, p):
        ## batch크기 구하기
        B= x.shape[0]

        ## 서로다른 패치크기로 conv연산진행
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        ## 벡터값에 PE더하기
        x1 = x1 + self.PE1
        x2 = x2 + self.PE2
        x3 = x3 + self.PE3

        ## Batch, Patch, Embedding으로 reshape
        x1 = x1.reshape(B, self.hidden_dim, -1).permute(0, 2, 1)
        x2 = x2.reshape(B, self.hidden_dim, -1).permute(0, 2, 1)
        x3 = x3.reshape(B, self.hidden_dim, -1).permute(0, 2, 1)

        return x1, x2, x3

class classfication_head(nn.Module):
    def __init__(self, dim, num_class):
        super().__init__()
        self.linear = nn.Linear(dim,num_class)
        self.softmax = nn.Softmax(-1)
    
    def forward(self,x):
        x = self.linear(x)

        patch_predictions = torch.argmax(x, dim=-1)
        abnormal_mask = (patch_predictions == 1).float() 

        abnormal_vectors = x * abnormal_mask.unsqueeze(-1)  # (batch_size, num_patches, num_classes)
        abnormal_sum = abnormal_vectors.sum(dim=1)  # 비정상 패치 벡터의 합 (batch_size, num_classes)
        abnormal_count = abnormal_mask.sum(dim=1, keepdim=True)  # 비정상 패치의 개수 (batch_size, 1)

        # 비정상 패치 벡터 평균 (비정상이 없으면 전체 평균 사용)
        abnormal_average = abnormal_sum / (abnormal_count + 1e-6)  # (batch_size, num_classes)
        normal_average = x.mean(dim=1)  # 전체 패치 벡터 평균 (batch_size, num_classes)

        # 최종 출력: 비정상이 있으면 비정상 평균, 없으면 전체 평균
        abnormal_exists = (abnormal_count > 0).float()  # (batch_size, 1)
        final_output = abnormal_exists * abnormal_average + (1 - abnormal_exists) * normal_average

        return final_output



class transcnn(nn.Module):
    def __init__(self, input_dim = 1, dim1 = 16, dim2 = 128, dim3 = 1024, len1=128, len2 = 32, len3 = 8, head = 4, layers = 3, task="classfication", num_class=2):
        super().__init__()
        ## value instance
        self.layers = layers
        self.task = task

        ## class instance
        self.stem = STEM(input_dim, dim1, dim2, dim3, len1, len2, len3)

        ## cross12 == 1->2  cross23 == 2->3
        self.cross12 = Cross_Fusion(head, len1*len1, len2*len2, dim1, dim2)
        self.cross23 = Cross_Fusion(head, len2*len2, len3*len3, dim2, dim3)

        self.cross32 = Cross_Fusion(head, len3*len3, len2*len2, dim3, dim2)
        self.cross21 = Cross_Fusion(head, len2*len2, len1*len1, dim2, dim1)

        self.upsample = nn.ConvTranspose2d(dim1, dim1, kernel_size=6, stride=4, padding=1)
        self.conv = nn.Conv2d(dim1, dim1, kernel_size=4, stride=4)
        self.seg = nn.Conv2d(dim1, num_class, kernel_size=1, stride=1)


        self.attention = nn.ModuleList(self_attention(head, len3*len3, dim3) for _ in range(layers) )

        if task == "classification":
            self.c_head = classfication_head(num_class)

        ## weight initialize
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, 0.0, 0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Parameter):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, p):
        x1,x2,x3 = self.stem(x,p)

        x_2 = self.cross12(x1,x2)
        x = self.cross23(x_2,x3)

        att_score = []
        for i in range(self.layers):
            x, attn = self.attention[i](x)
            att_score.append(attn)

        if self.task == "classification":
            output = self.c_head(x)
        elif self.task == "segmentation":
            h = self.cross32(x,x2)
            h = self.cross21(h,x1)
            h = self.upsample(h)
            h = self.conv(h)
            output = self.seg(h)
            

        return output, att_score


